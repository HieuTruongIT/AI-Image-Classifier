import { useCallback, useEffect, useMemo, useRef, useState } from "react";

// ─────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────
type TopK = { label: string; confidence: number };
type PredictOut = {
  label: string;
  confidence: number;
  top_k: TopK[];
  model_version: string;
  inference_ms?: number;
};

type ExplainOut = {
  label: string;
  confidence: number;
  heatmap_base64: string;
  model_version: string;
};

type ServerState = "unknown" | "online" | "waking" | "offline";
type Theme = "dark" | "light";

type HistoryItem = {
  id: string;
  file: File;
  previewUrl: string;
  result: PredictOut;
  latencyMs: number;
  timestamp: Date;
  heatmap?: string;
};

type SampleImage = {
  id: string;
  name: string;
  url: string;
  category: string;
};

const API = import.meta.env.VITE_API_URL as string;

// ─────────────────────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────────────────────
export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingExplain, setLoadingExplain] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictOut | null>(null);
  const [heatmap, setHeatmap] = useState<string | null>(null);
  const [server, setServer] = useState<ServerState>("unknown");
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [theme, setTheme] = useState<Theme>("dark");
  const [showSamples, setShowSamples] = useState(false);
  const [samples, setSamples] = useState<SampleImage[]>([]);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Theme effect
  useEffect(() => {
    document.documentElement.classList.toggle("light", theme === "light");
  }, [theme]);

  // Health check and fetch samples on mount
  useEffect(() => {
    const ctrl = new AbortController();
    const init = async () => {
      try {
        const r = await fetch(`${API}/health`, { signal: ctrl.signal });
        setServer(r.ok ? "online" : "waking");
        
        // Fetch sample images
        const samplesRes = await fetch(`${API}/samples`, { signal: ctrl.signal });
        if (samplesRes.ok) {
          const data = await samplesRes.json();
          setSamples(data.samples || []);
        }
      } catch {
        setServer("waking");
      }
    };
    init();
    return () => ctrl.abort();
  }, []);

  // Cleanup preview URL
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const onPick = useCallback((f: File) => {
    if (!f.type.startsWith("image")) {
      setError("Only images are allowed.");
      return;
    }
    if (f.size > 10 * 1024 * 1024) {
      setError("Max file size is 10MB.");
      return;
    }
    setError(null);
    setFile(f);
    setResult(null);
    setHeatmap(null);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(f);
    });
  }, []);

  const loadSampleImage = async (sample: SampleImage) => {
    setShowSamples(false);
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(sample.url);
      const blob = await response.blob();
      const file = new File([blob], `${sample.id}.jpg`, { type: "image/jpeg" });
      onPick(file);
    } catch {
      setError("Failed to load sample image. Try uploading your own.");
    } finally {
      setLoading(false);
    }
  };

  // Drag and drop handlers
  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        onPick(e.dataTransfer.files[0]);
      }
    },
    [onPick]
  );

  const predict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setHeatmap(null);
    setLatencyMs(null);

    const form = new FormData();
    form.append("file", file);

    const t0 = performance.now();
    try {
      const res = await fetch(`${API}/predict?top_k=5`, {
        method: "POST",
        body: form,
      });
      const t1 = performance.now();
      const ms = Math.round(t1 - t0);
      setLatencyMs(ms);

      if (!res.ok) {
        const text = await res.text();
        if (res.status === 413) throw new Error("File too large (over server limit).");
        if (res.status === 415) throw new Error("Unsupported file type.");
        if (res.status === 429) throw new Error("Rate limit exceeded. Please wait.");
        throw new Error(text || `Request failed (${res.status})`);
      }
      const data = (await res.json()) as PredictOut;
      setResult(data);
      setServer("online");

      // Add to history
      if (previewUrl) {
        setHistory((prev) => [
          {
            id: crypto.randomUUID(),
            file,
            previewUrl,
            result: data,
            latencyMs: ms,
            timestamp: new Date(),
          },
          ...prev.slice(0, 9),
        ]);
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Request failed.";
      setError(msg);
      if (server === "unknown") setServer("waking");
    } finally {
      setLoading(false);
    }
  };

  const explainPrediction = async () => {
    if (!file) return;
    setLoadingExplain(true);

    const form = new FormData();
    form.append("file", file);

    try {
      const res = await fetch(`${API}/explain?alpha=0.5`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) throw new Error("Failed to generate explanation");
      
      const data = (await res.json()) as ExplainOut;
      setHeatmap(data.heatmap_base64);
      setShowHeatmap(true);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Explanation failed.";
      setError(msg);
    } finally {
      setLoadingExplain(false);
    }
  };

  const exportResults = () => {
    if (!result) return;
    
    const exportData = {
      prediction: result,
      timestamp: new Date().toISOString(),
      file_name: file?.name,
      latency_ms: latencyMs,
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `visionscale-prediction-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const prettyTopK = useMemo(() => {
    if (!result) return [];
    return result.top_k.map((t) => ({ ...t, pct: Math.round(t.confidence * 100) }));
  }, [result]);

  const loadFromHistory = (item: HistoryItem) => {
    setFile(item.file);
    setPreviewUrl(item.previewUrl);
    setResult(item.result);
    setLatencyMs(item.latencyMs);
    setHeatmap(item.heatmap || null);
    setError(null);
  };

  const clearHistory = () => {
    history.forEach((h) => URL.revokeObjectURL(h.previewUrl));
    setHistory([]);
  };

  const themeClasses = theme === "dark" 
    ? "bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white"
    : "bg-gradient-to-br from-gray-50 via-white to-gray-100 text-gray-900";

  const cardClasses = theme === "dark"
    ? "bg-slate-800/50 border-slate-700/50"
    : "bg-white border-gray-200 shadow-sm";

  const inputClasses = theme === "dark"
    ? "border-slate-700 hover:border-slate-600 bg-slate-800/50"
    : "border-gray-300 hover:border-gray-400 bg-white";

  return (
    <div className={`min-h-screen ${themeClasses}`}>
      <div className="max-w-6xl mx-auto p-6">
        {/* Header */}
        <header className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center font-bold text-lg shadow-lg shadow-blue-500/25 text-white">
              V
            </div>
            <div>
              <h1 className={`text-2xl font-bold ${theme === "dark" ? "bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent" : "text-gray-900"}`}>
                VisionScale
              </h1>
              <p className={`text-xs ${theme === "dark" ? "text-slate-500" : "text-gray-500"}`}>
                Production ML Inference • 1000 Classes
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              className={`p-2 rounded-lg transition-colors ${theme === "dark" ? "bg-slate-800 hover:bg-slate-700" : "bg-gray-100 hover:bg-gray-200"}`}
              title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            >
              {theme === "dark" ? <SunIcon /> : <MoonIcon />}
            </button>
            <ServerBadge state={server} theme={theme} />
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Panel */}
          <div className="lg:col-span-2 space-y-6">
            {/* Sample Images Button */}
            <div className="flex gap-3">
              <button
                onClick={() => setShowSamples(!showSamples)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  theme === "dark"
                    ? "bg-slate-800 hover:bg-slate-700 text-slate-300"
                    : "bg-gray-100 hover:bg-gray-200 text-gray-700"
                }`}
              >
                {showSamples ? "Hide Samples" : "Try Sample Images"}
              </button>
              <a
                href={`${API}/docs`}
                target="_blank"
                rel="noopener noreferrer"
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  theme === "dark"
                    ? "bg-slate-800 hover:bg-slate-700 text-slate-300"
                    : "bg-gray-100 hover:bg-gray-200 text-gray-700"
                }`}
              >
                API Docs
              </a>
            </div>

            {/* Sample Images Grid */}
            {showSamples && samples.length > 0 && (
              <div className={`rounded-xl border p-4 ${cardClasses} animate-fadeIn`}>
                <h3 className={`text-sm font-medium mb-3 ${theme === "dark" ? "text-slate-400" : "text-gray-600"}`}>
                  Sample Images
                </h3>
                <div className="grid grid-cols-3 sm:grid-cols-6 gap-3">
                  {samples.map((sample) => (
                    <button
                      key={sample.id}
                      onClick={() => loadSampleImage(sample)}
                      className={`group relative rounded-lg overflow-hidden aspect-square transition-transform hover:scale-105 ${
                        theme === "dark" ? "ring-slate-600" : "ring-gray-300"
                      } hover:ring-2 ring-blue-500`}
                    >
                      <img
                        src={sample.url}
                        alt={sample.name}
                        className="w-full h-full object-cover"
                        loading="lazy"
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end p-2">
                        <span className="text-white text-xs font-medium truncate">
                          {sample.name}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Upload Zone */}
            <div
              className={`
                relative rounded-2xl border-2 border-dashed transition-all duration-300 p-8
                ${dragActive 
                  ? "border-blue-500 bg-blue-500/10 scale-[1.02]" 
                  : `${inputClasses}`
                }
              `}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={(e) => e.target.files && onPick(e.target.files[0])}
                className="hidden"
              />

              {previewUrl ? (
                <div className="space-y-4">
                  <div className="relative group flex justify-center gap-4">
                    <div className="relative">
                      <img
                        src={showHeatmap && heatmap ? `data:image/png;base64,${heatmap}` : previewUrl}
                        alt="Preview"
                        className="max-h-80 rounded-xl shadow-2xl object-contain transition-transform duration-300 group-hover:scale-[1.02]"
                      />
                      {heatmap && (
                        <button
                          onClick={() => setShowHeatmap(!showHeatmap)}
                          className={`absolute top-3 left-3 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                            showHeatmap
                              ? "bg-purple-600 text-white"
                              : "bg-black/60 backdrop-blur text-white hover:bg-black/80"
                          }`}
                        >
                          {showHeatmap ? "Show Original" : "Show Heatmap"}
                        </button>
                      )}
                    </div>
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="absolute bottom-3 right-3 px-3 py-1.5 rounded-lg bg-black/60 backdrop-blur text-xs text-white hover:bg-black/80 transition-colors"
                    >
                      Change Image
                    </button>
                  </div>
                  <p className={`text-center text-sm ${theme === "dark" ? "text-slate-400" : "text-gray-500"}`}>
                    {file?.name} • {file && (file.size / 1024).toFixed(1)} KB
                  </p>
                </div>
              ) : (
                <div
                  className="text-center cursor-pointer py-8"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <div className={`w-16 h-16 mx-auto mb-4 rounded-2xl flex items-center justify-center ${
                    theme === "dark" ? "bg-slate-700/50" : "bg-gray-100"
                  }`}>
                    <UploadIcon theme={theme} />
                  </div>
                  <p className={`text-lg font-medium mb-2 ${theme === "dark" ? "text-slate-300" : "text-gray-700"}`}>
                    Drop an image here, or click to browse
                  </p>
                  <p className={`text-sm ${theme === "dark" ? "text-slate-500" : "text-gray-500"}`}>
                    Supports JPG, PNG, WebP • Max 10MB
                  </p>
                </div>
              )}
            </div>

            {/* Action Buttons */}
            <div className="flex gap-3">
              <button
                onClick={predict}
                disabled={!file || loading}
                className={`
                  flex-1 py-4 rounded-xl font-semibold text-lg transition-all duration-300
                  ${!file || loading
                    ? `${theme === "dark" ? "bg-slate-700 text-slate-500" : "bg-gray-200 text-gray-400"} cursor-not-allowed`
                    : "bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 hover:scale-[1.02]"
                  }
                `}
              >
                {loading ? (
                  <span className="inline-flex items-center gap-3">
                    <Spinner /> Analyzing...
                  </span>
                ) : (
                  "Classify Image"
                )}
              </button>
              
              {result && (
                <button
                  onClick={explainPrediction}
                  disabled={loadingExplain}
                  className={`
                    px-6 py-4 rounded-xl font-semibold transition-all duration-300
                    ${loadingExplain
                      ? `${theme === "dark" ? "bg-slate-700 text-slate-500" : "bg-gray-200 text-gray-400"} cursor-not-allowed`
                      : `${theme === "dark" ? "bg-purple-600/20 text-purple-400 hover:bg-purple-600/30" : "bg-purple-100 text-purple-700 hover:bg-purple-200"}`
                    }
                  `}
                  title="Generate Grad-CAM heatmap"
                >
                  {loadingExplain ? <Spinner /> : <BrainIcon />}
                </button>
              )}
            </div>

            {/* Error */}
            {error && (
              <div className={`p-4 rounded-xl border animate-shake ${
                theme === "dark" 
                  ? "bg-red-500/10 border-red-500/20 text-red-400"
                  : "bg-red-50 border-red-200 text-red-600"
              }`}>
                {error}
              </div>
            )}

            {/* Results */}
            {result && (
              <div className={`rounded-2xl border overflow-hidden animate-fadeIn ${cardClasses}`}>
                <div className={`p-6 border-b ${theme === "dark" ? "border-slate-700/50" : "border-gray-200"}`}>
                  <div className="flex items-center justify-between mb-2">
                    <span className={`text-sm ${theme === "dark" ? "text-slate-400" : "text-gray-500"}`}>Prediction</span>
                    <div className="flex items-center gap-4">
                      {latencyMs !== null && (
                        <div className={`flex items-center gap-2 text-xs ${theme === "dark" ? "text-slate-500" : "text-gray-400"}`}>
                          <ClockIcon />
                          {latencyMs}ms
                        </div>
                      )}
                      <button
                        onClick={exportResults}
                        className={`text-xs px-2 py-1 rounded transition-colors ${
                          theme === "dark"
                            ? "bg-slate-700 hover:bg-slate-600 text-slate-300"
                            : "bg-gray-100 hover:bg-gray-200 text-gray-600"
                        }`}
                        title="Export as JSON"
                      >
                        Export
                      </button>
                    </div>
                  </div>
                  <div className="flex items-baseline gap-3">
                    <span className="text-3xl font-bold capitalize">{result.label}</span>
                    <span className="text-xl text-blue-400">
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  {result.confidence < 0.5 && (
                    <p className={`text-xs mt-2 ${theme === "dark" ? "text-amber-400" : "text-amber-600"}`}>
                      Low confidence - prediction may be uncertain
                    </p>
                  )}
                </div>

                <div className="p-6 space-y-4">
                  <div className={`text-sm mb-3 ${theme === "dark" ? "text-slate-400" : "text-gray-600"}`}>
                    Top Predictions
                  </div>
                  {prettyTopK.map((r, i) => (
                    <div key={i} className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="capitalize font-medium">{r.label}</span>
                        <span className={`tabular-nums ${theme === "dark" ? "text-slate-400" : "text-gray-500"}`}>
                          {r.pct}%
                        </span>
                      </div>
                      <div className={`h-2 rounded-full overflow-hidden ${theme === "dark" ? "bg-slate-700" : "bg-gray-200"}`}>
                        <div
                          className={`h-full rounded-full transition-all duration-700 ease-out ${
                            i === 0
                              ? "bg-gradient-to-r from-blue-500 to-purple-500"
                              : theme === "dark" ? "bg-slate-500" : "bg-gray-400"
                          }`}
                          style={{ width: `${r.pct}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                <div className={`px-6 py-3 text-xs flex justify-between ${
                  theme === "dark" ? "bg-slate-900/50 text-slate-500" : "bg-gray-50 text-gray-500"
                }`}>
                  <span>Model: {result.model_version}</span>
                  {heatmap && <span className="text-purple-400">Grad-CAM available</span>}
                </div>
              </div>
            )}
          </div>

          {/* Sidebar */}
          <div className="space-y-4">
            {/* History */}
            <div className="flex items-center justify-between">
              <h2 className={`text-lg font-semibold ${theme === "dark" ? "text-slate-300" : "text-gray-700"}`}>
                History
              </h2>
              {history.length > 0 && (
                <button
                  onClick={clearHistory}
                  className={`text-xs transition-colors ${theme === "dark" ? "text-slate-500 hover:text-slate-400" : "text-gray-400 hover:text-gray-600"}`}
                >
                  Clear all
                </button>
              )}
            </div>

            {history.length === 0 ? (
              <div className={`rounded-xl border p-8 text-center ${cardClasses}`}>
                <HistoryIcon theme={theme} />
                <p className={`text-sm mt-3 ${theme === "dark" ? "text-slate-500" : "text-gray-500"}`}>
                  Your prediction history will appear here
                </p>
              </div>
            ) : (
              <div className="space-y-3 max-h-[400px] overflow-y-auto pr-2 scrollbar-thin">
                {history.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => loadFromHistory(item)}
                    className={`w-full p-3 rounded-xl border transition-all duration-200 text-left group ${cardClasses} hover:scale-[1.02]`}
                  >
                    <div className="flex gap-3">
                      <img
                        src={item.previewUrl}
                        alt=""
                        className="w-14 h-14 rounded-lg object-cover flex-shrink-0"
                      />
                      <div className="flex-1 min-w-0">
                        <p className={`font-medium capitalize truncate group-hover:text-blue-400 transition-colors ${
                          theme === "dark" ? "text-slate-200" : "text-gray-800"
                        }`}>
                          {item.result.label}
                        </p>
                        <p className={`text-sm ${theme === "dark" ? "text-slate-500" : "text-gray-500"}`}>
                          {(item.result.confidence * 100).toFixed(0)}% • {item.latencyMs}ms
                        </p>
                        <p className={`text-xs mt-1 ${theme === "dark" ? "text-slate-600" : "text-gray-400"}`}>
                          {item.timestamp.toLocaleTimeString()}
                        </p>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            )}

            {/* Model Info Card */}
            <div className={`rounded-xl border p-4 ${cardClasses}`}>
              <h3 className={`text-sm font-medium mb-3 ${theme === "dark" ? "text-slate-400" : "text-gray-600"}`}>
                Model Info
              </h3>
              <div className="space-y-2 text-xs">
                {[
                  ["Architecture", "ResNet18"],
                  ["Dataset", "ImageNet-1K"],
                  ["Classes", "1,000"],
                  ["Format", "TorchScript"],
                  ["Inference", "CPU"],
                  ["Explainability", "Grad-CAM"],
                ].map(([label, value]) => (
                  <div key={label} className="flex justify-between">
                    <span className={theme === "dark" ? "text-slate-500" : "text-gray-500"}>{label}</span>
                    <span className={theme === "dark" ? "text-slate-300" : "text-gray-700"}>{value}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Keyboard Shortcuts */}
            <div className={`rounded-xl border p-4 ${cardClasses}`}>
              <h3 className={`text-sm font-medium mb-3 ${theme === "dark" ? "text-slate-400" : "text-gray-600"}`}>
                Keyboard Shortcuts
              </h3>
              <div className="space-y-2 text-xs">
                {[
                  ["Ctrl + V", "Paste image"],
                  ["Enter", "Classify"],
                  ["E", "Explain (Grad-CAM)"],
                  ["T", "Toggle theme"],
                ].map(([key, action]) => (
                  <div key={key} className="flex justify-between">
                    <kbd className={`px-1.5 py-0.5 rounded font-mono ${
                      theme === "dark" ? "bg-slate-700 text-slate-300" : "bg-gray-100 text-gray-700"
                    }`}>{key}</kbd>
                    <span className={theme === "dark" ? "text-slate-500" : "text-gray-500"}>{action}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className={`mt-12 pt-6 border-t text-center text-xs ${
          theme === "dark" ? "border-slate-800 text-slate-600" : "border-gray-200 text-gray-400"
        }`}>
          <p>VisionScale • Production ML Inference Service</p>
          <p className="mt-1">ResNet18 • TorchScript • FastAPI • React • Prometheus</p>
        </footer>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Components
// ─────────────────────────────────────────────────────────────
function Spinner() {
  return (
    <span
      className="inline-block h-5 w-5 animate-spin rounded-full border-2 border-current border-r-transparent"
      aria-hidden
    />
  );
}

function ServerBadge({ state, theme }: { state: ServerState; theme: Theme }) {
  const configs = {
    dark: {
      online: { color: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20", dot: "bg-emerald-500" },
      waking: { color: "bg-amber-500/10 text-amber-400 border-amber-500/20", dot: "bg-amber-500 animate-pulse" },
      offline: { color: "bg-red-500/10 text-red-400 border-red-500/20", dot: "bg-red-500" },
      unknown: { color: "bg-slate-500/10 text-slate-400 border-slate-500/20", dot: "bg-slate-500" },
    },
    light: {
      online: { color: "bg-emerald-50 text-emerald-600 border-emerald-200", dot: "bg-emerald-500" },
      waking: { color: "bg-amber-50 text-amber-600 border-amber-200", dot: "bg-amber-500 animate-pulse" },
      offline: { color: "bg-red-50 text-red-600 border-red-200", dot: "bg-red-500" },
      unknown: { color: "bg-gray-50 text-gray-600 border-gray-200", dot: "bg-gray-400" },
    },
  };
  
  const labels = { online: "Online", waking: "Waking", offline: "Offline", unknown: "Checking..." };
  const { color, dot } = configs[theme][state];

  return (
    <span className={`inline-flex items-center gap-2 text-xs px-3 py-1.5 rounded-full border ${color}`}>
      <span className={`w-2 h-2 rounded-full ${dot}`} />
      {labels[state]}
    </span>
  );
}

function UploadIcon({ theme }: { theme: Theme }) {
  return (
    <svg className={`w-8 h-8 ${theme === "dark" ? "text-slate-400" : "text-gray-400"}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
    </svg>
  );
}

function ClockIcon() {
  return (
    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  );
}

function HistoryIcon({ theme }: { theme: Theme }) {
  return (
    <svg className={`w-12 h-12 mx-auto ${theme === "dark" ? "text-slate-700" : "text-gray-300"}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  );
}

function SunIcon() {
  return (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
    </svg>
  );
}

function BrainIcon() {
  return (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
    </svg>
  );
}
