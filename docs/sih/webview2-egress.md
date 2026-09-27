# WebView2's own traffic to Microsoft, and how ARJUN stops it

The problem statement asks ARJUN to "show, through logs or a visible network
monitor, that no external calls are made at any point"
([`ps-26117-official.md`](ps-26117-official.md)). ARJUN's own code already kept
that promise. The component that draws its window did not.

## What was found

ARJUN's window is Microsoft Edge WebView2, a runtime installed with Windows and
not written by us. We launched ARJUN **online** on WebView2 runtime 153.0.4234.48
and recorded every TCP connection opened by ARJUN's process tree, polling every
100 ms. We also turned on Chromium's net-log. Before the fix, a single launch
produced:

| Contacted | From | Why |
| --- | --- | --- |
| `edge.microsoft.com/componentupdater/api/v1/update` | WebView2 network service, about 60 s after launch | Component updates |
| `wpad` (DNS lookup) | WebView2 network service | Proxy auto-discovery |
| `substrate.office.com` (IPv6 `2603:1046:c04:14xx::2`) | WebView2 **browser process**, within seconds of launch | `oneauth.dll` fetching the Windows account's profile and picture (`/profileb2/v2.0/me/V1Profile`, `/imageB2/v1.0/me/image`) |

The third row does not go through Chromium's network stack. OneAuth uses its own
HTTP client, so it never appears in the net-log, and no network switch reaches
it. We identified it from three pieces of evidence:
- the owning process: the browser process, not the network service;
- its loaded modules: `oneauth.dll` plus the WAM account extension;
- the address: it resolves in the same pool as `substrate.office.com`, and
  `oneauth.dll` contains that URL.

None of this came from the page. ARJUN's CSP permits only `self` and its IPC
endpoints, so the frontend cannot open an external connection.

It had gone unnoticed for a reason. ARJUN's in-app observer asked Windows only
about `sarathi.exe`'s own process ID, and the WebView2 browser is a separate
child process.

## The fix

**1. Launch switches** (`src-tauri/tauri.conf.json`, main window
`additionalBrowserArgs`):

| Switch | Stops |
| --- | --- |
| `--disable-features=…,msOneAuthWAM` | Loading OneAuth, and with it the `substrate.office.com` profile fetch. Found by bisecting 29 identity features over repeated launches; this one alone removes the connection, and `oneauth.dll` is no longer loaded. |
| `--disable-component-update` | The component updater poll |
| `--no-proxy-server` | The `wpad` lookup |
| `--disable-background-networking` | Variations seed and other background fetches |
| `--host-resolver-rules="MAP * ~NOTFOUND, EXCLUDE localhost, EXCLUDE *.localhost, EXCLUDE 127.0.0.1"` | Everything else sent through Chromium's network stack. Every hostname except this machine fails inside the WebView **before a DNS query is made**, whatever feature asked for it. |

Setting `additionalBrowserArgs` replaces wry's defaults
(`msWebOOUI,msPdfOOUI,msSmartScreenProtection`, autoplay), so they are
repeated in the config.

`sovereignty::webview` pins every switch with its reason. Its tests fail if a
switch is removed, or if the resolver rule exempts anything that is not
loopback.

**2. The observer watches the whole process tree**
(`src-tauri/src/sovereignty/observer.rs`). That means `sarathi.exe` and every
process it started: the WebView2 browser and its helpers, llama-server, the
Python sidecars and the agent runtime. Each connection names its process, and
the Audit & Network page and the health panel say which process it was.

Windows reuses process IDs, so a process that is older than its recorded parent
is not counted as ARJUN's. When a start time can't be read, the process is kept:
an extra process can only add connections to the report, never hide one.

## How it was verified

The probe launched ARJUN online and recorded every TCP connection opened by
ARJUN's process tree, with the owning process's Chromium role. The runs were
made with `WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS`, before the switches were
built in:

| Launch | External connections |
| --- | --- |
| wry defaults only | 2: `substrate.office.com` (browser process), `edge.microsoft.com` (network service) |
| plus `--disable-component-update --no-proxy-server` | 1: `substrate.office.com` |
| plus `msOneAuthWAM` disabled | 0 |
| the full set above, 100 s, UI exercised over CDP | 0, and the net-log names no host other than `*.localhost` |

The same probe was then run on the deployed build, with the switches built
in, with ARJUN online the whole time:

| Session | Length | External connections |
| --- | --- | --- |
| Idle at the sign-in screen | 4 h 00 min | 0 |
| Signed in; demo steps 3, 4 and 5 (scan OCR, SOP comparison, Word approval note); Audit & Network page; closed normally and watched 30 s more | about 15 min each, three sessions | 0 in each |

Over the four idle hours the net-log shows the runtime tried a
DNS-over-HTTPS probe to `dns.google` 393 times. Each attempt was resolved to
`~NOTFOUND` inside the WebView by the resolver rule, so no DNS query and no
connection left the machine. That is the rule doing work that no switch
targeted.

After the build, the deployed `sarathi.exe`'s WebView2 browser process was
confirmed to carry every switch. Debugging flags passed through
`WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS` merge with the switches; they do not
replace them. Only a switch of the same name is overridden.

## Limits

- **Runtime updates.** Feature names such as `msOneAuthWAM` belong to the
  runtime and can change when it updates. The resolver rule covers the Chromium
  network stack whatever a feature is called. The browser process's own HTTP
  clients (OneAuth, and the `oneds.dll` telemetry client) are covered only by
  feature switches. Measure again after a runtime update. An air-gapped
  deployment should pin a **Fixed Version** WebView2 runtime so this cannot
  change underneath it.
- **Observer coverage.** The observer sees TCP only. It can miss a connection
  that opens and closes between two polls, and it does not see UDP. It
  corroborates the broker; it is not a packet capture.
- **Test length.** The longest online soak is four hours idle. Telemetry on a
  schedule longer than that is not ruled out by it.
