# One Line to Video

Project repository: [ximizhou/one-line-to-video](https://github.com/ximizhou/one-line-to-video).

An agentic service that turns a short brief into a **vertical (9:16) 30/60-second AI video**: optional web research → script → reflective revision → storyboard/style bible → per-shot AI video → optional TTS narration → MP4 assembly.

Built on **LangGraph**, provider adapters, **DeepSeek or Gemini** for structured text generation, **MiniMax-H3** (or a future provider) for scene clips, optional **IndexTTS**, and **ffmpeg** for assembly.

```text
POST /storyboard ─▶ job_id
        │
        ▼
  research (optional) → script_writer → reflection → designer → video_gen → assembler
      web sources        DeepSeek/LLM    revise loop    storyboard  H3/provider  TTS + MP4
                                                               └─ bounded GPU pool

  Postgres: job/artifact metadata only
  ARTIFACT_ROOT: research.json, script.json, shotlist.json, style_bible.md,
                 clip_NN.mp4, narration.wav (if TTS), storyboard.mp4
GET  /storyboard/{id}         ─▶ status + stage timeline + artifacts
GET  /storyboard/{id}/logs    ─▶ per-stage log lines
GET  /storyboard/{id}/events  ─▶ SSE progress/log stream
GET  /storyboard/{id}/video   ─▶ final mp4
POST /storyboard/{id}/resume  ─▶ resume idempotently from saved artifacts
```

**Pipeline graph** (`research` is conditionally included when `RESEARCH_ENABLED=true`):

```mermaid
graph TD
  start -->|research on| research --> script_writer
  start -->|research off| script_writer
  script_writer --> reflection --> designer --> video_gen --> assembler --> pipeline_end([Done])
```

**Narrative consistency:** `designer` creates a style bible and self-contained shot prompts. The active pipeline generates video clips directly; legacy image/motion stages remain available for compatibility.

**Offline development:** select Mock providers to test the graph without external services. Real LLM keys are read from environment variables and never sent by the UI.


## Local visual workspace

The repository includes a static visual workspace served by FastAPI. It lets you choose the LLM/model, research source, video provider/model, H3 text/image mode, GPU scheduling, and TTS provider from the UI. Seedance remains an adapter placeholder until its API protocol is supplied.

### UI-only preview (no database or model services)

Run the following from the repository root in PowerShell. If `.venv` has not
been created yet, install the Python dependencies first:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

To inspect the interface and provider/model selectors without PostgreSQL,
API keys, an SSH tunnel, a video GPU, or a TTS service:

```powershell
$env:USE_MOCK_PROVIDERS = "true"
$env:LLM_PROVIDER = "mock"
$env:VIDEO_PROVIDER = "mock"
$env:RESEARCH_ENABLED = "false"
$env:ENABLE_TTS = "false"

.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --lifespan off
```

Keep the terminal open and visit **http://127.0.0.1:8000/ui/**.
Press **Ctrl+C** to stop the preview. If port 8000 is busy, use `--port 8001`
and open **http://127.0.0.1:8001/ui/** instead.

> **Preview only:** `--lifespan off` skips the database-dependent startup hooks;
> it does not initialize or repair the database. Do not submit generation jobs
> or load existing jobs in this mode. Job submission, progress/log streaming,
> and video generation require a healthy, initialized database. Do not use this
> flag for normal operation or deployment.

### Full application (task submission and generation)

Stop the preview, then use a fresh PowerShell terminal so its Mock-only
settings do not override your normal provider configuration. Run from the
repository root with the Python dependencies installed. Docker must be running.

These commands use the local development database defined in
`docker-compose.yml`. For an existing PostgreSQL instance, set `DATABASE_URL`
to your own connection URL and skip the Docker command instead.

```powershell
$env:DATABASE_URL = "postgresql+asyncpg://storyboard:storyboard@127.0.0.1:5432/storyboard"

docker compose up -d --wait
if ($LASTEXITCODE -ne 0) { throw "PostgreSQL did not become healthy." }

.\.venv\Scripts\python.exe -m alembic upgrade head
if ($LASTEXITCODE -ne 0) { throw "Database migration failed." }

.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/ui/** after application startup completes.
`--wait` requires a Docker Compose version that supports it and waits for the
Postgres healthcheck before migrations run. If it is unavailable, run
`docker compose up -d`, then check `docker compose ps` until Postgres is
`healthy` before running Alembic and Uvicorn.

Mock providers can exercise the workflow without external model services;
real generation additionally requires the selected providers' API keys and/or
reachable video/TTS endpoints. Keys stay in the server environment, not the UI.

The UI can:

- submit a one-sentence prompt;
- show the LangGraph stage timeline;
- stream stage logs over SSE;
- display token usage and job status;
- preview the final MP4 and each generated scene clip;
- choose the LLM/model and research website;
- choose the video provider/model, text/image mode, GPU pool (`GPU 2`, `GPU 3`, or `GPU 2 + 3`), concurrency, TTS provider, and (when IndexTTS is reachable) the narration voice.

The active media graph is now:

```text
script_writer → reflection → designer → video_gen → assembler
```

`video_gen` calls the provider boundary in `app/adapters/video_model.py`. The first
real implementation is `h3_workbench`: it defaults to H3 FL2VA pure text-to-video,
submits a job on the selected provider GPU, polls every item, and downloads the
resulting WebM before the local assembler creates the final MP4. Switching
`VIDEO_INPUT_MODE=i2v` enables the optional seed-frame path. Per-shot dispatch round-robins through `VIDEO_GPU_POOL` while `VIDEO_MAX_CONCURRENCY` bounds submitted work. Text-to-video providers such as Seedance can implement the same adapter contract without changing the graph.

TTS is disabled by default. Set `ENABLE_TTS=true` and choose `TTS_PROVIDER=indextts`
for the existing IndexTTS workbench (or `gemini` for the old cloud adapter). The IndexTTS
client follows its explicit create -> start -> poll -> download protocol. When the workbench
`/api/voices` endpoint is reachable, the UI loads built-in and saved voices and sends the
selected `tts_voice` with the job; leaving it on auto uses the first available workbench voice.
For Chinese history/science narration, prefer a saved voice marked as a natural/reading style
(for example, the prepared server has a saved voice named “黄轩朗读”).

Provider-video assembly has one timeline: clips are trimmed or padded with the final frame,
never looped from the beginning, and the same scene holds drive subtitles and narration. The
assembler writes `subtitles.srt` and burns the captions into the MP4, so a short provider clip
will not replay its action with a silent second pass. Keep `VIDEO_FRAMES=0` to derive the H3
frame count from `VIDEO_CLIP_SECONDS`; a manually low frame count can otherwise make the
provider return a clip shorter than the requested hold.

### Remote/local video workbench

The project only needs the workbench HTTP endpoint; deployment and authentication remain outside this repository. Keep the workbench bound to a private interface and expose it locally through your own approved tunnel or network policy.

Set `VIDEO_PROVIDER=h3_workbench`, `VIDEO_MODEL=fl2va`, `VIDEO_INPUT_MODE=t2v`, and `VIDEO_BASE_URL` to the local endpoint that you control. Use `VIDEO_GPU_POOL=2,3` and `VIDEO_MAX_CONCURRENCY=2` only when those GPU IDs are actually available to your process. `VIDEO_QUALITY=uhd` requests the highest configured H3 tier; `VIDEO_FRAMES` controls generated motion length. Keep inline `VIDEO_ENHANCE=false` while H3 is resident: SeedVR2 can exceed remaining VRAM when run beside H3 on the same card. Use `VIDEO_INPUT_MODE=i2v` only when intentionally conditioning on a seed image.

The adapter expects `GET /api/health`, `GET /api/models`, `POST /api/upload`, `POST /api/job`, `GET /api/jobs/{id}`, and an output download route. A top-level `done` is accepted only after every `items[*]` entry has no error.

---

## Prerequisites

- Python 3.11+
- Docker (for local Postgres) — or your own Postgres
- `ffmpeg` on your PATH (`brew install ffmpeg` on macOS)

## Setup

```bash
# 1. Clone + enter
git clone https://github.com/ximizhou/one-line-to-video.git
cd one-line-to-video

# 2. Virtualenv + install
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 3. Env
cp .env.example .env
#   - For this demo, expose the DeepSeek key to the server as the environment variable dsh.
#   - The UI selects providers/models but never accepts or displays API keys.

# 4. Start Postgres and wait for its healthcheck
docker compose up -d --wait

# 5. Create schema
alembic upgrade head

# 6. Run the service
uvicorn app.main:app --reload
```

> **Note on `--reload`:** it restarts the process on file changes. Because jobs run in-process, a
> restart mid-job is detected on startup and that job is marked `interrupted` (not left hanging).

## Local testing (curl)

```bash
# Kick off a job
curl -s -XPOST localhost:8000/storyboard \
  -H 'content-type: application/json' \
  -d '{"prompt":"a lonely robot finds a flower","duration":30}'
# -> {"job_id":"<uuid>","status":"queued"}

# Poll status (artifacts include any degraded frames + warnings)
curl -s localhost:8000/storyboard/<job_id> | python -m json.tool

# Download the video once status is completed / completed_with_warnings
curl -o out.mp4 localhost:8000/storyboard/<job_id>/video

# Resume an interrupted/failed job (e.g. after a crash mid-generation).
# Idempotent: finished frames are reused, only the missing ones are regenerated.
curl -s -XPOST localhost:8000/storyboard/<job_id>/resume
# -> 202 (resuming) | 409 (already running / already finished) | 404 (unknown)
```

### Live status & per-stage logs (front-end ready)

While a job runs you can watch *which stage is executing*, which scene/video stage is running, and
the exact log lines for each stage — by polling or by subscribing to a live stream.

```bash
# Collapsed view: GET /{id} now carries `current_stage` + a `stages[]` summary
curl -s localhost:8000/storyboard/<job_id> | python -m json.tool
#   "current_stage": "video_gen",
#   "stages": [ {"name":"script_writer","status":"succeeded", ...},
#               {"name":"video_gen","status":"running",
#                "progress_current":3,"progress_total":10}, ... ]

# Expanded view: one stage's log lines (paginate with ?after=<last id>)
curl -s 'localhost:8000/storyboard/<job_id>/logs?stage=video_gen' | python -m json.tool

# Live stream (Server-Sent Events) — the front end's subscribe path
curl -N localhost:8000/storyboard/<job_id>/events
```

The SSE stream emits four event types and closes itself when the job finishes:

```
event: status   data: {"status":"running","current_stage":"video_gen"}
event: stage    data: [ {"name":"video_gen","status":"running",...}, ... ]
event: log      id: 142   data: {"id":142,"stage":"video_gen","level":"info","message":"H3 job submitted: ... gpu=2", ...}
event: done     data: {"status":"completed_with_warnings"}
```

`log` events carry an `id:` — the cursor. A browser `EventSource` auto-reconnects and sends
`Last-Event-ID`; the server replays only the lines after that id (no dupes, no gaps). Captured logs
include the agents' own progress lines **and** incidental adapter logs (e.g. the image model's
re-roll retries), each attributed to the stage that produced it.

> Logs are best-effort (a bounded queue + a single async drainer): under an extreme flood the
> oldest lines are dropped and the drop is surfaced as a synthetic line — never silent. The
> *stage* status (which stage is running) is written directly and is always reliable.

Job statuses: `queued → running → completed | completed_with_warnings | failed | interrupted`.
`completed_with_warnings` means some frames needed the reliability fallback (see `warnings` +
per-artifact `status: degraded`): the image adapter **re-rolls** empty/no-image responses, and any
frame that still fails is **seamlessly filled from its nearest good neighbour** (no gap, no jarring
card) — a styled placeholder appears only in the rare all-frames-failed case. If the process dies
mid-job it becomes `interrupted` — `POST /resume` continues it instead of restarting from zero.

### Resilience: idempotency + checkpointer

Every node is **load-or-produce** (reuses its artifact if present), and the graph is compiled with
a LangGraph checkpointer (Postgres in prod, sqlite locally) keyed by `job_id`. Together: a resumed
job reuses the same script/shotlist/style-ref and only regenerates missing frames — no divergent
re-script, no duplicate artifacts, no re-billing finished frames.

## Running tests

```bash
pytest          # hermetic: mock providers + sqlite, no ffmpeg/Postgres/API key needed
```

## Motion engine integration

Motion remains off by default. Set `MOTION_ENGINE_ENABLED=true` to insert the
optional motion stage and call `storyboard_engine` at `MOTION_ENGINE_URL`.
`MOTION_IN_FILM=true` splices returned clips into the final film; otherwise clips
are UI-only. `MOTION_ENGINE_ENABLED=false` preserves the original pipeline.

The integration accepts both v1 and v2 engine responses. With Motion Engine v2,
the backend writes `motion_manifest.json` beside the job artifacts and returns
optional per-scene fields (`motion_kind`, `renderer`, `model`, `motion_reason`,
`pipeline_version`) plus aggregate `motion_coverage`. These additions require no
database migration. Under `generative_all`, composited/Ken Burns scenes are
marked degraded and the completed job carries a visible warning.

The assembler always removes clip audio. Existing TTS narration is the only final
soundtrack.

## Provider configuration notes

The active script/reflection path supports DeepSeek, Gemini, OpenAI-compatible providers, and Mock. Set provider/model names in `.env`; keys stay server-side. For the DeepSeek demo, `DEEPSEEK_API_KEY` may be omitted when the server process already has `dsh` in its environment.

The older Gemini image-generation and Ken Burns configuration remains in the repository for legacy pipeline/tests, but it is not used by the active `video_gen` graph.

## Configuration (`.env`)

| Key | Purpose |
|---|---|
| `GEMINI_API_KEY` | AI Studio key. Blank → mock mode. |
| `GEMINI_TEXT_MODEL` / `GEMINI_IMAGE_MODEL` | Model ids. Recommended: best Gemini-3 text + Nano Banana 2 image (resolve exact ids with your key). |
| `USE_MOCK_PROVIDERS` | Force mock mode even with a key. |
| `LLM_PROVIDER` / `LLM_MODEL` | Structured script/reflection provider. DeepSeek reads `dsh` from the server environment; do not put keys in the UI or source. |
| `RESEARCH_ENABLED` / `RESEARCH_PROVIDER` | Optional source collection and selected search website/provider. |
| `VIDEO_PROVIDER` / `VIDEO_MODEL` / `VIDEO_INPUT_MODE` | AI video provider, model, and T2V/I2V mode. |
| `VIDEO_GPU_POOL` / `VIDEO_MAX_CONCURRENCY` | Round-robin GPU assignment and bounded per-shot concurrency. |
| `VIDEO_QUALITY` / `VIDEO_FRAMES` / `VIDEO_STEPS` | H3 resolution, generated frame count, and sampling steps. |
| `VIDEO_ENHANCE` / `VIDEO_ENHANCE_MODEL` | Optional post-generation enhancement. Keep off while H3 is resident on the same GPU unless scheduling enhancement after H3 releases VRAM. |
| `ENABLE_TTS` / `TTS_PROVIDER` / `TTS_VOICE_ID` | Optional IndexTTS or Gemini narration. |
| `DATABASE_URL` | Async SQLAlchemy URL (asyncpg). |
| `ARTIFACT_ROOT` | Where images/videos are written. |
| `IMAGE_GEN_MAX_RETRIES` | Re-rolls on empty/no-image responses before neighbour-fill (default `5`). |
| `IMAGE_ASPECT_RATIO` | Shape requested from the image model (default `9:16` vertical → no pillarbox). |
| `IMAGE_MAX_CONCURRENCY` / `IMAGE_RPM_LIMIT` | Bounded concurrency + shared RPM ceiling for image calls. |
| `ENABLE_MOTION` / `CROSSFADE_SECONDS` | Ken Burns zoom + crossfades (default on). Off → plain concat of stills. |
| `MOTION_ENGINE_ENABLED` / `MOTION_ENGINE_URL` | Enable the optional external motion stage and set its URL. |
| `MOTION_IN_FILM` | Splice per-scene motion clips into the assembled film. |
| `MOTION_CLIP_SECONDS` / `MOTION_STAGE_BUDGET_SECONDS` | Requested source clip length and total motion-stage budget. |
| `MOTION_MAX_CONCURRENCY` | Maximum concurrent scene requests to the motion engine. |
| `SECONDS_PER_FRAME` | Seconds each frame is held → drives frame count (default `2.0`). |
| `CAPTION_FONT_PATH` | Optional `.ttf` for nicer subtitles (default: Pillow's scalable font). |
| `IMAGE_SEED` | Optional fixed seed for the image model (default off; best-effort, ref image carries consistency). |
| `ENABLE_LOG_STREAMING` | Per-stage status + log streaming (default on). Off → `/events` 503s; pull endpoints still serve rows. |
| `LOG_POLL_INTERVAL_SECONDS` | How often the SSE stream polls the DB (default `0.75`). |
| `LOG_QUEUE_MAX` / `LOG_DRAIN_BATCH` / `LOG_CAPTURE_LEVEL` | Bounded log-capture queue size, drainer batch size, min captured level. |

## Quality evals (manual)

Mocked unit tests prove the pipeline *wires up*; they can't judge whether the script reads well or
the style holds. The golden-prompt harness runs a few prompts against the **real** models and dumps
every artifact for human review, with cheap heuristics (schema, frame count, caption length):

```bash
python -m evals.run            # real models (needs GEMINI_API_KEY)
python -m evals.run --mock     # offline smoke run (placeholder frames)
python -m evals.run --limit 2  # first N prompts only
# Outputs land under evals/output/<timestamp>/<slug>/
```

## Project layout

See `app/` — `api/` (routes), `graph/` (LangGraph state, wiring, checkpointer), `agents/`
(research, script_writer, reflection, designer, video_gen, assembler), `adapters/` (research,
llm, video, tts — provider boundaries and retries), `artifacts/` (filesystem store, atomic writes), `db/`
(models, repository, session), `observability/` (per-stage reporter + the log-capture bridge:
contextvar, bounded queue, async drainer), `workers/` (background runner). `evals/` holds the
manual golden-prompt harness.

**Phase 1** (vertical slice) + **Phase 2** (quality layer) + **Phase 3** (polish: vertical 9:16,
subtitle captions, reliable frames via re-roll + neighbour-fill, Ken Burns motion + crossfades,
bounded concurrency under the rate limit, optional TTS voiceover) + **per-stage logging & live
status streaming** (SSE + pull endpoints; auto-captured per-stage logs) are implemented in the legacy image workflow. The default graph now uses direct AI video generation; legacy image/motion adapters are retained for compatibility.

## Motion engine (optional — per-scene "live photo" clips)

The separate `storyboard_engine` service (not included in this repository; default port 8090) turns each frame into
a short **motion clip** (depth-based 2.5D parallax; Google Veo for a couple of hero frames). When
enabled, a `motion` stage runs between `image_gen` and `assembler`, animating every frame; the
clips appear per-scene in the UI (`scenes[].video_url`) **and** replace the Ken Burns still-zooms
inside `storyboard.mp4`.

**Legacy integration — not part of the default AI-video graph.** The adapter is gated by `MOTION_ENGINE_ENABLED` (default `false`); it is retained for compatibility with the earlier image/motion workflow.

- `MOTION_ENGINE_ENABLED=true` — call the engine (must be running on `MOTION_ENGINE_URL`).
- `MOTION_IN_FILM=true` — splice clips into the final mp4 (needs `ENABLE_MOTION=true`; per-scene
  fallback to the Ken Burns zoom when a clip is missing). Set `false` to keep clips in the UI only.
- **Never fails:** engine unreachable / slow / a clip errors → that scene silently keeps its still
  Ken Burns; the job always completes.
- Do NOT confuse `ENABLE_MOTION` (local Ken Burns zoom vs concat) with `MOTION_ENGINE_ENABLED`
  (call the external engine). See `.env.example`.

Run it:
```bash
cd ../storyboard_engine && uvicorn app.main:app --port 8090   # see its README
# then start this backend with MOTION_ENGINE_ENABLED=true
```

## License and attribution

MIT licensed; see [LICENSE](LICENSE). This project is based on [Adithya-nat/storyboard_agent](https://github.com/Adithya-nat/storyboard_agent). The original copyright notice is retained; One Line to Video adds the research, configurable video/TTS providers, GPU dispatch, and visual-workspace extensions.

## Keeping deployment information private

Keep real API keys in environment variables or a local `.env` file; `.env.example` contains only a configuration template. Do not commit private server addresses, credentials, logs, job databases, or generated media. Use the configurable workbench URLs for your own deployment rather than hard-coding infrastructure into the adapters.
