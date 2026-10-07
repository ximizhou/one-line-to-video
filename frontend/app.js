const $ = (id) => document.getElementById(id);
const API_BASE = window.location.port === "3000" ? "http://localhost:8000" : "";
const apiUrl = (path) => `${API_BASE}${path}`;
const state = { jobId: null, timer: null, source: null, providers: null };

const statusText = { queued: "排队中", running: "生成中", completed: "已完成", completed_with_warnings: "已完成（有提示）", failed: "失败", interrupted: "已中断" };
const stageText = { script_writer: "脚本", reflection: "反思", designer: "分镜", video_gen: "视频生成", assembler: "合成" };

function addLog(stage, message, level = "info") {
  const box = $("logs");
  const empty = box.querySelector(".log-empty"); if (empty) empty.remove();
  const row = document.createElement("div"); row.className = `log-line ${level === "error" ? "error" : ""}`;
  row.innerHTML = `<b>[${stage || "system"}]</b> ${escapeHtml(message)}`;
  box.appendChild(row); box.scrollTop = box.scrollHeight;
}
function escapeHtml(value) { return String(value).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"}[c])); }
function setStatus(status) { const el = $("jobStatus"); el.textContent = statusText[status] || status || "等待输入"; el.className = `job-status ${status || ""}`; $("metricStatus").textContent = statusText[status] || status || "—"; }

function renderPipeline(stages) {
  const box = $("pipeline");
  if (!stages?.length) { box.innerHTML = '<div class="empty-line">创建一个主题后，这里会显示每个 LangGraph 节点的状态。</div>'; return; }
  box.innerHTML = stages.map((stage, i) => {
    const stateText = stage.status === "succeeded" ? "完成" : stage.status === "running" ? `${stage.progress_current || 0}/${stage.progress_total || "…"}` : stage.status === "failed" ? "失败" : "等待";
    return `${i ? '<span class="stage-arrow">→</span>' : ''}<div class="stage ${stage.status}"><div class="stage-name">${stageText[stage.name] || stage.name}</div><div class="stage-state">${stateText}</div></div>`;
  }).join("");
}

async function refreshJob() {
  if (!state.jobId) return;
  try {
    const response = await fetch(apiUrl(`/storyboard/${state.jobId}`));
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "读取任务失败");
    setStatus(data.status); renderPipeline(data.stages);
    $("jobId").textContent = data.job_id;
    $("metricStage").textContent = stageText[data.current_stage] || data.current_stage || "—";
    $("metricTokens").textContent = data.total_tokens ?? 0;
    if (["completed", "completed_with_warnings", "failed", "interrupted"].includes(data.status)) {
      clearInterval(state.timer); state.timer = null;
      if (data.status !== "failed" && data.status !== "interrupted") await loadResult();
      $("createBtn").disabled = false;
    }
  } catch (error) { addLog("ui", error.message, "error"); }
}

async function loadResult() {
  const response = await fetch(apiUrl(`/storyboard/${state.jobId}/storyboard`));
  if (!response.ok) return;
  const data = await response.json();
  $("resultMeta").textContent = `${data.duration}s · ${data.scenes?.length || 0} 个镜头`;
  const preview = $("preview"); preview.src = data.video_url || ""; preview.hidden = !data.video_url; $("resultEmpty").style.display = data.video_url ? "none" : "flex";
  $("scenes").innerHTML = (data.scenes || []).map((scene) => `<div class="scene-card">${scene.video_url ? `<video src="${scene.video_url}" controls preload="metadata"></video>` : '<div class="scene-placeholder">clip missing</div>'}<span>镜头 ${scene.order} · ${escapeHtml(scene.narration || "无旁白")}</span></div>`).join("");
}

function connectEvents() {
  if (state.source) state.source.close();
  state.source = new EventSource(apiUrl(`/storyboard/${state.jobId}/events`));
  state.source.addEventListener("log", (event) => { try { const item = JSON.parse(event.data); addLog(item.stage, item.message, item.level); } catch (_) {} });
  state.source.addEventListener("done", () => state.source.close());
  state.source.onerror = () => { state.source.close(); };
}

async function createJob() {
  const prompt = $("prompt").value.trim(); if (prompt.length < 3) return addLog("ui", "请先输入至少 3 个字符的主题", "error");
  $("createBtn").disabled = true; $("resultEmpty").style.display = "flex"; $("preview").hidden = true; $("scenes").innerHTML = ""; $("logs").innerHTML = '<div class="log-empty">任务已提交，等待日志…</div>';
  try {
    const response = await fetch(apiUrl("/storyboard"), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
      prompt,
      duration: Number($("duration").value),
      llm_provider: $("llmProvider").value || null,
      llm_model: $("llmModel").value || null,
      research_provider: $("researchProvider").value || null,
      research_enabled: $("researchEnabled").value === "true",
      video_provider: $("videoProvider").value || null,
      video_model: $("videoModel").value || null,
      video_input_mode: $("videoInputMode").value || "t2v",
      video_gpu: Number(($("videoGpuPool").value || "2").split(",")[0]),
      video_gpu_pool: $("videoGpuPool").value || null,
      video_max_concurrency: Number($("videoConcurrency").value || "1"),
      tts_provider: $("ttsProvider").value || "none",
      tts_voice: $("ttsVoice").value || null,
    }) });
    const data = await response.json(); if (!response.ok) throw new Error(data.detail || "创建任务失败");
    state.jobId = data.job_id; $("jobId").textContent = data.job_id; setStatus(data.status); addLog("system", `任务 ${data.job_id} 已创建`); connectEvents(); await refreshJob(); state.timer = setInterval(refreshJob, 1500);
  } catch (error) { $("createBtn").disabled = false; addLog("ui", error.message, "error"); }
}

document.querySelectorAll(".suggestions button").forEach((button) => button.addEventListener("click", () => { $("prompt").value = button.dataset.prompt; }));
$("createBtn").addEventListener("click", createJob);
$("clearLogs").addEventListener("click", () => { $("logs").innerHTML = '<div class="log-empty">日志已清空。</div>'; });
$("loadLatest").addEventListener("click", () => addLog("ui", "请从任务 URL 或数据库中选择一个 job_id；当前页面不会自动猜测任务。"));
loadProviders();


function syncInputMode() {
  const provider = (state.providers?.video || []).find((item) => item.id === $("videoProvider").value);
  const model = (provider?.models || []).find((item) => item.id === $("videoModel").value);
  const allowed = model?.input_modes || (provider?.id === "h3_workbench" ? ["t2v", "i2v"] : ["t2v"]);
  [...$("videoInputMode").options].forEach((option) => { option.disabled = !allowed.includes(option.value); });
  if (!allowed.includes($("videoInputMode").value)) $("videoInputMode").value = allowed[0] || "t2v";
}

function populateVideoModels(providerId, preferredModel) {
  const profile = (state.providers?.video || []).find((item) => item.id === providerId) || state.providers?.video?.[0];
  const modelSelect = $("videoModel");
  modelSelect.innerHTML = (profile?.models || []).map((model) => `<option value="${escapeHtml(model.id)}">${escapeHtml(model.label || model.id)}</option>`).join("");
  if (preferredModel && [...modelSelect.options].some((option) => option.value === preferredModel)) modelSelect.value = preferredModel;
  syncInputMode();
  $("providerHint").textContent = profile?.hint || (profile?.configured === false ? "该 provider 尚未配置，先用 Mock 验证流程。" : profile?.requires_image ? "该 provider 会为每个分镜生成一个 seed frame，再调用图生视频。" : "可切换 provider，不需要改 LangGraph 节点。");
}

function populateLlmModels(providerId, preferredModel) {
  const profile = (state.providers?.llm || []).find((item) => item.id === providerId) || state.providers?.llm?.[0];
  const select = $("llmModel");
  select.innerHTML = (profile?.models || []).map((model) => `<option value="${escapeHtml(model.id)}">${escapeHtml(model.label || model.id)}</option>`).join("");
  if (preferredModel && [...select.options].some((option) => option.value === preferredModel)) select.value = preferredModel;
}

async function loadProviders() {
  try {
    const response = await fetch(apiUrl("/storyboard/providers"));
    if (!response.ok) throw new Error("读取 provider 列表失败");
    state.providers = await response.json();
    const defaults = state.providers.defaults || {};
    const llmSelect = $("llmProvider");
    llmSelect.innerHTML = (state.providers.llm || []).map((profile) => `<option value="${escapeHtml(profile.id)}">${escapeHtml(profile.label)}</option>`).join("");
    llmSelect.value = defaults.llm_provider || llmSelect.options[0]?.value || "mock";
    populateLlmModels(llmSelect.value, defaults.llm_model);
    llmSelect.addEventListener("change", () => populateLlmModels(llmSelect.value));
    const researchSelect = $("researchProvider");
    researchSelect.innerHTML = (state.providers.research || []).map((profile) => `<option value="${escapeHtml(profile.id)}">${escapeHtml(profile.label)}</option>`).join("");
    researchSelect.value = defaults.research_provider || "duckduckgo";
    $("researchEnabled").value = defaults.research_enabled ? "true" : "false";
    const videoSelect = $("videoProvider");
    videoSelect.innerHTML = (state.providers.video || []).map((profile) => `<option value="${escapeHtml(profile.id)}">${escapeHtml(profile.label)}</option>`).join("");
    videoSelect.value = defaults.video_provider || videoSelect.options[0]?.value || "mock";
    populateVideoModels(videoSelect.value, defaults.video_model);
    videoSelect.addEventListener("change", () => populateVideoModels(videoSelect.value));
    $("videoModel").addEventListener("change", syncInputMode);
    if (defaults.video_input_mode) $("videoInputMode").value = defaults.video_input_mode;
    if (defaults.video_gpu_pool) $("videoGpuPool").value = String(defaults.video_gpu_pool);
    else if (defaults.video_gpu != null) $("videoGpuPool").value = String(defaults.video_gpu);
    if (defaults.video_max_concurrency != null) $("videoConcurrency").value = String(defaults.video_max_concurrency);
    const ttsSelect = $("ttsProvider");
    ttsSelect.innerHTML = (state.providers.tts || []).map((profile) => `<option value="${escapeHtml(profile.id)}">${escapeHtml(profile.label)}</option>`).join("");
    ttsSelect.value = defaults.tts_provider || "none";
    const voiceSelect = $("ttsVoice");
    const voices = state.providers.tts_voices || [];
    voiceSelect.innerHTML = `<option value="">自动选择（服务端默认）</option>` + voices.map((voice) => {
      const detail = [voice.description, voice.language].filter(Boolean).join(" · ");
      const label = detail ? `${voice.label || voice.id}（${detail}）` : (voice.label || voice.id);
      return `<option value="${escapeHtml(voice.id)}">${escapeHtml(label)}</option>`;
    }).join("");
    if (defaults.tts_voice_id && [...voiceSelect.options].some((option) => option.value === defaults.tts_voice_id)) {
      voiceSelect.value = defaults.tts_voice_id;
    }
  } catch (error) {
    $("providerHint").textContent = "provider 列表不可用，回退到 Mock。";
    addLog("ui", error.message, "error");
  }
}
