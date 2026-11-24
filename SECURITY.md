# Security Policy

## Reporting a vulnerability

Email adrianwedd@gmail.com with subject "SECURITY: afterwords".
You'll get an acknowledgement within 72 hours. Please don't open public
issues for security reports.

## Threat model (short version)

The server is designed for localhost-only use on a single-user Mac. The
clone/reload/delete endpoints require an explicit --allow-clone opt-in and
are forced onto 127.0.0.1. Loopback binds reject Host headers that are not
a loopback name (DNS rebinding), and every request a browser marks as coming
from another site (a foreign or `null` Origin, or a Sec-Fetch-Site other
than same-origin/none) is refused with 403 before its body is read.
Non-loopback binds require --bind-public and are at your own risk. The /tmp queue/lock convention assumes a
single-user machine.

## Supported versions

The latest tagged release.
