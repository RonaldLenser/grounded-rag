# Deploying Grounded

Production setup: **Google Cloud Run** (the app, `us-east4`, Virginia) + **Neon** (managed
PostgreSQL with pgvector, `aws-us-east-1`, Virginia). Same region family = low latency.

Commands below are PowerShell (Windows). Secrets are typed at a hidden prompt and go straight
into Google Secret Manager: they are never written to files, logs, or shell history.

## 0. Hidden prompt helper (paste once per PowerShell window)

Never paste a secret onto the command line, and never use plain `Read-Host`: it **echoes**
what you paste, so the secret ends up on screen (and in any screenshot). Use this instead:

```powershell
function Read-Secret([string]$Prompt) {
    $secure = Read-Host -AsSecureString $Prompt          # shows **** while you paste
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try { [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
}
```

Functions don't survive closing the window: paste it again in each new window.

## Architecture

```
Browser ──HTTPS──> Cloud Run (container from Dockerfile, 0-2 instances, scale to zero)
                     │  secrets injected as env vars from Secret Manager
                     ├──> Neon Postgres + pgvector (direct connection, SSL)
                     ├──> OpenAI embeddings API
                     └──> Anthropic Claude API
```

## Cost and abuse protection

| Control | Setting | Why |
|---|---|---|
| Per-IP rate limit | 10 questions/min, 30 searches/min | Stops one client looping requests |
| Global daily cap | 300 LLM calls per UTC day, reserved atomically in Postgres before each call | Bounds worst-case spend across all instances; failed calls count too |
| Relevance gate | No LLM call for off-topic questions | Off-topic spam costs $0 |
| Writes | Admin API key only | The public can ask; only you can upload/delete |
| Instances | min 0, max 2 | $0 when idle; caps runaway scaling |

Typical costs on free tiers: Cloud Run, Neon, Secret Manager and Artifact Registry stay at or
near $0 for a demo. Each answered question is roughly $0.003-0.01 of Claude usage.

## 1. Database (Neon)

1. Sign up at https://neon.tech (GitHub login works).
2. Create a project: name `grounded-rag`, Postgres **17**, region **AWS US East 1 (N. Virginia)**.
3. Click **Connect**. Turn **Connection pooling OFF** and copy the connection string.
   - Pooled URLs (host contains `-pooler`) break asyncpg's prepared statements; the app
     refuses them with a clear error.
   - Paste the string exactly as Neon shows it. The app converts `postgresql://...?sslmode=require`
     to the async driver format itself.
4. Create the schema from your machine (session-only variable, gone when the window closes):

```powershell
$env:DATABASE_URL = Read-Secret "Paste Neon connection string"
uv run alembic upgrade head
Remove-Item Env:DATABASE_URL
```

## 2. Google Cloud project (one time)

```powershell
winget install --id Google.CloudSDK -e      # then CLOSE and reopen PowerShell
gcloud init                                  # log in; create a new project when asked
```

Pick a globally unique project ID, e.g. `grounded-rag-rl1234`. Then:

1. Link billing: https://console.cloud.google.com/billing (Cloud Run requires a billing account,
   even within the free tier).
2. Set a budget alert (Billing -> Budgets & alerts), e.g. **$10/month**, so surprises email you.
3. Enable the services and give the runtime service account the roles it needs:

```powershell
$project = gcloud config get-value project
$number  = gcloud projects describe $project --format="value(projectNumber)"
$sa      = "$number-compute@developer.gserviceaccount.com"

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com

# Read secrets at runtime; build images from source.
gcloud projects add-iam-policy-binding $project --member="serviceAccount:$sa" --role="roles/secretmanager.secretAccessor"
gcloud projects add-iam-policy-binding $project --member="serviceAccount:$sa" --role="roles/run.builder"
```

## 3. Secrets (one time)

Use a **new** admin key for production (`[guid]::NewGuid().ToString("N")`) and save it in your
password manager. Separate production API keys are recommended, so a leaked dev key can be
revoked without taking the demo down.

```powershell
function New-GroundedSecret([string]$Name) {
    Read-Secret "Paste value for $Name" | gcloud secrets create $Name --data-file=-
}

New-GroundedSecret DATABASE_URL        # the Neon string from step 1
New-GroundedSecret OPENAI_API_KEY
New-GroundedSecret ANTHROPIC_API_KEY
New-GroundedSecret ADMIN_API_KEY       # the NEW production admin key
```

(PowerShell adds a trailing newline when piping; the app strips surrounding whitespace from
all settings, so that can't corrupt a key.)

### Rotating a secret (e.g. after a leak)

1. Create the new value at the source first (Neon: Roles -> Reset password; OpenAI/Anthropic:
   new key). The old value must stop working, or rotation achieved nothing.
2. Store it and restart the app on it (no rebuild needed):

```powershell
Read-Secret "Paste NEW value for DATABASE_URL" | gcloud secrets versions add DATABASE_URL --data-file=-
gcloud run services update grounded --region us-east4 --update-secrets DATABASE_URL=DATABASE_URL:latest
```

3. Check the app: `https://<service-url>/ready` must return `{"status":"ready"}`.
4. Only then disable the old version(s): `gcloud secrets versions list DATABASE_URL`, then
   `gcloud secrets versions disable <N> --secret=DATABASE_URL`.

Disabling before step 3 can take the site down if the new value is wrong.

## 4. Deploy (repeat for every release)

If the release adds a database migration, run it against Neon **first** (step 1.4 command).
Migrations are additive, so the running version keeps working with the new schema, while the
new version would fail without it.

From the repo root, on an up-to-date `main`:

```powershell
gcloud run deploy grounded --source . --region us-east4 --allow-unauthenticated `
  --min-instances 0 --max-instances 2 --memory 1Gi --cpu 1 --concurrency 20 --timeout 300 `
  --set-env-vars "APP_ENV=prod,LOG_LEVEL=INFO,TRUSTED_PROXY_HOPS=1" `
  --set-secrets "DATABASE_URL=DATABASE_URL:latest,OPENAI_API_KEY=OPENAI_API_KEY:latest,ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest,ADMIN_API_KEY=ADMIN_API_KEY:latest"
```

- `--source .` uploads the repo (minus `.gcloudignore`), Cloud Build builds the `Dockerfile`,
  and Cloud Run runs it. First run asks to create an Artifact Registry repository: answer `Y`.
- `APP_ENV=prod` turns on JSON logs and makes the app refuse to start if any key is missing.
- `TRUSTED_PROXY_HOPS=1`: the client IP is the last `X-Forwarded-For` entry, appended by
  Google's front end. Verified by the spoofing check in step 5.
- The command prints the service URL: `https://grounded-<hash>.us-east4.run.app`.

## 5. Seed the demo and verify

```powershell
$url = "https://grounded-<hash>.us-east4.run.app"     # from the deploy output
$env:ADMIN_API_KEY = Read-Secret "Production admin key"
uv run python scripts/seed_demo.py $url
uv run python scripts/smoke_test.py $url --check-rate-limit
Remove-Item Env:ADMIN_API_KEY
```

Expect `ALL CHECKS PASSED`, including **"rate limit can't be bypassed with spoofed
X-Forwarded-For"**. If that one fails, the proxy-hop assumption is wrong for this platform:
do not leave the demo public until it's fixed.

## Operating it

- Logs: Cloud Console -> Cloud Run -> `grounded` -> **Logs** (JSON logs; filter by
  `jsonPayload.request_id` to follow one request).
- Usage: `query_logs` table (status, tokens, latency per question) and `llm_daily_usage`
  (LLM calls reserved per UTC day).
- Data retention: `query_logs` stores the raw question text with no automatic expiry. It is
  deleted only when its collection is deleted. See "Data retention" in DESIGN.md.
- Cold starts: with `min-instances 0` the first request after idle takes a few seconds.
  A real product would set `--min-instances 1` (costs money while idle).
