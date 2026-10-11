# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.2.x   | :white_check_mark: |
| < 1.2   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in VisionScale, please report it responsibly:

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. Open a private security advisory via GitHub's Security tab
3. Or email the maintainer directly through GitHub

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Fix Timeline**: Depends on severity

## Security Best Practices

When deploying VisionScale:

1. **API Keys**: Always set `AUTH_ENABLED=true` in production
2. **Rate Limiting**: Keep `RATE_LIMIT_ENABLED=true`
3. **HTTPS**: Always use HTTPS in production
4. **Secrets**: Never commit secrets to the repository
5. **Updates**: Keep dependencies updated

## Known Security Considerations

- The default Grafana password in `docker-compose.yml` is for local development only
- API key authentication is disabled by default (`AUTH_ENABLED=false`)
- Rate limiting is enabled by default
