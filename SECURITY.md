# Security Policy

## Supported Version

Security fixes are applied to the latest version on the `main` branch.

## Reporting a Vulnerability

Do not open a public issue containing credentials, biometric data, private images, or
working exploit details.

Use GitHub's private vulnerability reporting for this repository:

`https://github.com/a6rahamjr/VeriPresence/security/advisories/new`

Include the affected version, reproduction steps, expected impact, and any suggested
mitigation. Reports will be acknowledged after review.

## Sensitive Data

The repository intentionally excludes:

- `.env` files
- enrollment images
- trained model artifacts
- experiment runs
- SQLite attendance records

Operators must provide their own access control, encryption, retention rules, and consent
processes for biometric data.
