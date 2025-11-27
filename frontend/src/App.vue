<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
import { UploadFilled, VideoPlay, Download, RefreshRight, Plus, Loading, Check, Warning, Setting, Link } from '@element-plus/icons-vue'
import dayjs from 'dayjs'
import axios from 'axios'

// --- Configuration ---
const API_BASE_URL = '/api/v1' 
const POLL_INTERVAL = 2000 

// --- State ---
const isUploading = ref(false)
const showUploader = ref(true)
const timer = ref(null)
const uploadRef = ref(null) // 用于引用 upload 组件以清理文件
const selectedModel = ref('lama') // 新增：当前选择的模型

// Sora Downloader State
const soraUrl = ref('')
const isDownloading = ref(false)
const isLoggingIn = ref(false)

// Data State aligned with Backend Models
const queueSummary = ref({
  is_busy: false,
  queue_length: 0,
  total_active: 0
})

const currentTaskId = ref(null)
const currentTaskResult = ref(null) 
const waitingQueue = ref([]) 

// --- API Interactions ---

// 1. 获取队列状态
const fetchQueueStatus = async () => {
  try {
    const { data } = await axios.get(`${API_BASE_URL}/get_queue_status`)
    queueSummary.value = data.summary
    waitingQueue.value = data.waiting_queue
    const newCurrentTaskId = data.current_task_id
    currentTaskId.value = newCurrentTaskId

    if (newCurrentTaskId) {
      fetchCurrentTaskResult(newCurrentTaskId)
    } else {
      currentTaskResult.value = null
    }
  } catch (error) {
    console.error('Failed to fetch queue status:', error)
  }
}

// 2. 获取特定任务结果
const fetchCurrentTaskResult = async (taskId) => {
  try {
    const { data } = await axios.get(`${API_BASE_URL}/get_results`, {
      params: { remove_task_id: taskId }
    })
    currentTaskResult.value = {
      id: taskId,
      status: data.status,
      percentage: data.percentage,
      video_path: 'Processing...', 
      created_at: null, 
      download_url: data.download_url
    }
  } catch (error) {
    console.error('Failed to fetch task result:', error)
  }
}

// 3. 提交任务
const handleUploadChange = async (file) => {
  isUploading.value = true
  const formData = new FormData()
  formData.append('video', file.raw)

  try {
    // 修改：使用 selectedModel 的值
    await axios.post(`${API_BASE_URL}/submit_remove_task`, formData, {
      params: { cleaner_type: selectedModel.value },
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    
    ElMessage.success({ message: `Task submitted: ${file.name}`, plain: true })
    
    // 修改：不再隐藏上传框，而是刷新队列并清理当前文件，允许继续上传
    // showUploader.value = false 
    if (uploadRef.value) {
      uploadRef.value.clearFiles()
    }
    fetchQueueStatus() 
  } catch (error) {
    ElMessage.error({ message: `Upload failed: ${error.message}`, plain: true })
  } finally {
    isUploading.value = false
  }
}

// Sora Actions
const handleSoraLogin = async () => {
  isLoggingIn.value = true
  try {
    await axios.post(`${API_BASE_URL}/sora/login`)
    ElMessage.success({ message: 'Browser launched for Sora login', plain: true })
  } catch (error) {
    ElMessage.error({ message: `Login launch failed: ${error.message}`, plain: true })
  } finally {
    isLoggingIn.value = false
  }
}

const handleSoraDownload = async () => {
  if (!soraUrl.value) {
    ElMessage.warning('Please enter a Sora video URL')
    return
  }

  isDownloading.value = true
  try {
    await axios.post(`${API_BASE_URL}/sora/download`, { url: soraUrl.value })
    ElMessage.success({ message: 'Video downloaded and queued!', plain: true })
    soraUrl.value = ''
    fetchQueueStatus()
  } catch (error) {
    ElMessage.error({ message: `Download failed: ${error.response?.data?.detail || error.message}`, plain: true })
  } finally {
    isDownloading.value = false
  }
}

// --- Computed Logic ---
const tableData = computed(() => {
  const list = []
  if (currentTaskId.value && currentTaskResult.value) {
    list.push({
      id: currentTaskId.value,
      status: currentTaskResult.value.status, 
      percentage: currentTaskResult.value.percentage,
      video_path: currentTaskResult.value.video_path, 
      created_at: null, 
      is_current: true
    })
  }
  if (waitingQueue.value && waitingQueue.value.length > 0) {
    waitingQueue.value.forEach(task => {
      list.push({
        id: task.id,
        status: task.status, 
        percentage: task.percentage,
        video_path: task.video_path,
        created_at: task.created_at,
        is_current: false
      })
    })
  }
  return list
})

const stats = computed(() => {
  return {
    totalActive: queueSummary.value.total_active, 
    queueLength: queueSummary.value.queue_length, 
    isBusy: queueSummary.value.is_busy ? 1 : 0    
  }
})

// --- Helpers ---
const formatDate = (dateStr) => {
  if (!dateStr) return '-'
  return dayjs(dateStr).format('MMM D, HH:mm')
}

const getDownloadUrl = (taskId) => {
  return `${API_BASE_URL}/download/${taskId}`
}

const getStatusConfig = (status) => {
  const map = {
    'FINISHED': { type: 'success', label: 'Ready', icon: Check, bg: 'pill-green' },
    'PROCESSING': { type: 'primary', label: 'Processing', icon: Loading, bg: 'pill-blue' },
    'QUEUED': { type: 'info', label: 'Queued', icon: null, bg: 'pill-gray' },
    'UPLOADING': { type: 'warning', label: 'Uploading', icon: Loading, bg: 'pill-gray' },
    'ERROR': { type: 'danger', label: 'Failed', icon: Warning, bg: 'pill-red' }
  }
  return map[status] || { type: 'info', label: status, bg: 'pill-gray' }
}

// --- Lifecycle ---
onMounted(() => {
  fetchQueueStatus()
  timer.value = setInterval(fetchQueueStatus, POLL_INTERVAL)
})

onUnmounted(() => {
  if (timer.value) clearInterval(timer.value)
})
</script>

<template>
  <div class="oa-page">
    <header class="oa-header">
      <div class="oa-header-inner">
        <div class="oa-brand">
          <div class="oa-dot" :class="{ 'oa-dot-busy': queueSummary.is_busy }" />
          <span class="oa-title">Video Tasks</span>
        </div>
        <div class="oa-actions">
          <el-button
            class="oa-primary-btn"
            :icon="Plus"
            @click="showUploader = !showUploader"
          >
            {{ showUploader ? 'Hide upload' : 'New task' }}
          </el-button>
        </div>
      </div>
    </header>

    <main class="oa-container">
      <section class="oa-stats">
        <div class="oa-stat-card">
          <div class="oa-stat-label">System Status</div>
          <div class="oa-stat-value">
            {{ queueSummary.is_busy ? 'Busy' : 'Idle' }}
          </div>
        </div>

        <div class="oa-stat-card">
          <div class="oa-stat-label">Queue Length</div>
          <div class="oa-stat-value">{{ stats.queueLength }}</div>
        </div>

        <div class="oa-stat-card">
          <div class="oa-stat-label">Total Active</div>
          <div class="oa-stat-value">{{ stats.totalActive }}</div>
        </div>
      </section>

      <transition name="el-fade-in-linear">
        <section v-if="showUploader" class="oa-upload-section">

          <!-- Sora Downloader Block -->
          <div class="oa-sora-downloader">
            <h3 class="oa-section-title" style="margin-bottom: 12px;">Sora Downloader</h3>
            <div class="oa-sora-login">
              <el-button
                class="oa-login-btn"
                :loading="isLoggingIn"
                @click="handleSoraLogin"
              >
                <el-icon style="margin-right: 6px;"><Link /></el-icon>
                Login to Sora (Recommended)
              </el-button>
            </div>

            <div class="oa-sora-input-group">
              <el-input
                v-model="soraUrl"
                placeholder="Paste Sora video URL here..."
                class="oa-sora-input"
                clearable
              >
                <template #prefix>
                   <el-icon><Link /></el-icon>
                </template>
              </el-input>
              <el-button
                type="primary"
                class="oa-download-btn"
                :loading="isDownloading"
                @click="handleSoraDownload"
              >
                Download
              </el-button>
            </div>
          </div>

          <div class="oa-controls">
            <span class="oa-control-label">Model:</span>
            <el-radio-group v-model="selectedModel" size="small" class="oa-radio-group">
              <el-radio-button label="lama">Lama (Fast)</el-radio-button>
              <el-radio-button label="e2fgvi_hq">E2FGVI (High Quality)</el-radio-button>
            </el-radio-group>
          </div>

          <el-upload
            ref="uploadRef"
            class="oa-uploader"
            drag
            action="#"
            :auto-upload="false"
            :on-change="handleUploadChange"
            :show-file-list="false"
            :disabled="isUploading"
          >
            <div class="oa-upload-inner">
              <el-icon class="oa-upload-icon" v-if="!isUploading"><UploadFilled /></el-icon>
              <el-icon class="oa-upload-icon is-loading" v-else><Loading /></el-icon>
              
              <div class="oa-upload-text">
                <span v-if="!isUploading">
                  <span class="oa-upload-strong">Click to upload</span>
                  or drag video
                </span>
                <span v-else>Uploading to server...</span>
              </div>
              <div class="oa-upload-hint">MP4, MOV, AVI · Max 500MB</div>
            </div>
          </el-upload>
        </section>
      </transition>

      <section class="oa-table">
        <div class="oa-table-head">
          <h3 class="oa-section-title">Current & Queue</h3>
          <el-button
            :icon="RefreshRight"
            text
            size="small"
            class="oa-refresh"
            @click="fetchQueueStatus"
          >
            Refresh
          </el-button>
        </div>

        <el-table
          :data="tableData"
          class="oa-el-table"
          :row-style="{ height: '68px' }"
          empty-text="No active tasks"
        >
          <el-table-column label="Video Info" min-width="320">
            <template #default="{ row }">
              <div class="oa-file-cell">
                <div class="oa-file-icon">
                  <el-icon><VideoPlay /></el-icon>
                </div>
                <div class="oa-file-info">
                  <div class="oa-file-name">{{ row.video_path || `Task: ${row.id.substring(0,8)}...` }}</div>
                  <div class="oa-file-meta">{{ row.id }}</div>
                </div>
              </div>
            </template>
          </el-table-column>

          <el-table-column label="Status" width="150">
            <template #default="{ row }">
              <div class="oa-status-pill" :class="getStatusConfig(row.status).bg">
                <span class="oa-status-dot"></span>
                {{ getStatusConfig(row.status).label }}
              </div>
            </template>
          </el-table-column>

          <el-table-column label="Progress" width="220">
            <template #default="{ row }">
              <div class="oa-progress">
                <el-progress
                  :percentage="row.percentage"
                  :show-text="false"
                  :stroke-width="4"
                  :color="row.status === 'ERROR' ? '#ef4444' : '#10a37f'"
                  :indeterminate="row.status === 'PROCESSING' && row.percentage === 0"
                  class="oa-progress-bar"
                />
                <span class="oa-progress-text">
                  {{ row.percentage }}%
                </span>
              </div>
            </template>
          </el-table-column>

          <el-table-column label="Created At" width="170" align="right">
            <template #default="{ row }">
              <span class="oa-date">{{ formatDate(row.created_at) }}</span>
            </template>
          </el-table-column>

          <el-table-column width="64" align="center">
            <template #default="{ row }">
              <a 
                v-if="row.status === 'FINISHED'"
                :href="getDownloadUrl(row.id)" 
                target="_blank"
                class="oa-download-link"
              >
                <el-button link class="oa-download">
                  <el-icon><Download /></el-icon>
                </el-button>
              </a>
            </template>
          </el-table-column>
        </el-table>
      </section>
    </main>
  </div>
</template>

<style scoped>
/* ---------------------------
   OpenAI-like Design Tokens
---------------------------- */
:root {
  --oa-bg: #ffffff;
  --oa-surface: #f7f7f8;
  --oa-surface-2: #fbfbfc;
  --oa-border: #e6e6e9;
  --oa-text: #0b0c0e;
  --oa-text-2: #5f6368;
  --oa-text-3: #8a8f98;
  --oa-green: #10a37f;
  --oa-black: #0b0c0e;
  --oa-radius-lg: 14px;
  --oa-radius-md: 10px;
  --oa-shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
}

/* Page + container */
.oa-page {
  min-height: 100vh;
  background: var(--oa-bg);
  color: var(--oa-text);
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Inter, sans-serif;
}

.oa-container {
  max-width: 1040px;
  margin: 0 auto;
  padding: 28px 24px 56px;
}

/* Header */
.oa-header {
  position: sticky;
  top: 0;
  z-index: 5;
  background: rgba(255,255,255,0.9);
  backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--oa-border);
}

.oa-header-inner {
  max-width: 1040px;
  margin: 0 auto;
  padding: 14px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.oa-brand {
  display: flex;
  align-items: center;
  gap: 10px;
}

.oa-dot {
  width: 10px;
  height: 10px;
  background: #ccc; 
  border-radius: 999px;
  transition: background 0.3s ease;
}
.oa-dot-busy {
  background: var(--oa-green);
  box-shadow: 0 0 8px rgba(16, 163, 127, 0.4);
}

.oa-title {
  font-size: 15px;
  font-weight: 600;
  letter-spacing: 0.1px;
}

/* Primary button */
.oa-primary-btn {
  background: var(--oa-black) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 999px !important;
  padding: 8px 14px !important;
  font-weight: 600;
  box-shadow: var(--oa-shadow-sm);
}
.oa-primary-btn:hover { background: #000 !important; }

/* Stats */
.oa-stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr); 
  gap: 14px;
  margin-top: 20px;
  margin-bottom: 22px;
}

.oa-stat-card {
  background: var(--oa-surface);
  border: 1px solid var(--oa-border);
  border-radius: var(--oa-radius-lg);
  padding: 18px;
  box-shadow: var(--oa-shadow-sm);
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.oa-stat-label {
  font-size: 12px;
  color: var(--oa-text-2);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.oa-stat-value {
  font-size: 26px;
  font-weight: 700;
  letter-spacing: -0.02em;
}

/* Upload & Controls */
.oa-upload-section {
  margin-top: 8px;
  margin-bottom: 26px;
}

.oa-controls {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.oa-control-label {
  font-size: 13px;
  font-weight: 600;
  color: var(--oa-text-2);
}

/* Sora Downloader Styles */
.oa-sora-downloader {
  background: var(--oa-surface);
  border: 1px solid var(--oa-border);
  border-radius: var(--oa-radius-lg);
  padding: 16px;
  margin-bottom: 20px;
}

.oa-sora-login {
  display: flex;
  justify-content: center;
  margin-bottom: 16px;
}

.oa-login-btn {
  background: transparent;
  border: 1px solid var(--oa-border);
  color: var(--oa-text);
  border-radius: 8px;
  font-weight: 600;
}
.oa-login-btn:hover {
  background: var(--oa-surface-2);
  border-color: var(--oa-text-3);
  color: var(--oa-text);
}

.oa-sora-input-group {
  display: flex;
  gap: 10px;
}

.oa-sora-input {
  flex: 1;
}
.oa-sora-input :deep(.el-input__wrapper) {
  border-radius: 8px;
  background: #fff;
  box-shadow: 0 0 0 1px var(--oa-border) inset;
}
.oa-sora-input :deep(.el-input__wrapper:hover) {
  box-shadow: 0 0 0 1px var(--oa-text-3) inset;
}
.oa-sora-input :deep(.el-input__wrapper.is-focus) {
  box-shadow: 0 0 0 1px var(--oa-green) inset;
}

.oa-download-btn {
  background: linear-gradient(90deg, #a855f7 0%, #ec4899 100%) !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 600;
  padding: 8px 20px !important;
}
.oa-download-btn:hover {
  opacity: 0.9;
}

/* Customizing Radio Button to look cleaner */
.oa-radio-group :deep(.el-radio-button__inner) {
  border-radius: 6px !important;
  border: 1px solid var(--oa-border);
  box-shadow: none !important;
  margin-right: 8px;
  padding: 8px 16px;
  font-weight: 500;
  background: var(--oa-surface);
  color: var(--oa-text);
}
.oa-radio-group :deep(.el-radio-button__original-radio:checked + .el-radio-button__inner) {
  background-color: var(--oa-black);
  border-color: var(--oa-black);
  color: #fff;
  box-shadow: none;
}
.oa-radio-group :deep(.el-radio-button:first-child .el-radio-button__inner) {
  border-left: 1px solid var(--oa-border);
}

.oa-uploader :deep(.el-upload-dragger) {
  height: 128px;
  border: 1.5px dashed var(--oa-border);
  background: var(--oa-surface-2);
  border-radius: var(--oa-radius-lg);
  transition: all 0.2s ease;
}
.oa-uploader :deep(.el-upload-dragger:hover) {
  border-color: var(--oa-green);
  background: #f3fbf8;
}

.oa-upload-inner {
  height: 100%;
  display: grid;
  place-content: center;
  gap: 6px;
  text-align: center;
}

.oa-upload-icon { font-size: 22px; color: var(--oa-text-3); }
.oa-upload-text { font-size: 14px; color: var(--oa-text-2); }
.oa-upload-strong { color: var(--oa-green); font-weight: 700; }
.oa-upload-hint { font-size: 12px; color: var(--oa-text-3); }
.is-loading { animation: rotating 2s linear infinite; }

/* Table Section */
.oa-table {
  background: var(--oa-bg);
  border: 1px solid var(--oa-border);
  border-radius: var(--oa-radius-lg);
  padding: 14px 12px 6px;
  box-shadow: var(--oa-shadow-sm);
}

.oa-table-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 8px 12px;
}

.oa-section-title {
  margin: 0;
  font-size: 16px;
  font-weight: 700;
  letter-spacing: -0.01em;
}

.oa-refresh { color: var(--oa-text-2) !important; }

/* Element Plus table overrides */
.oa-el-table {
  --el-table-border-color: transparent;
  --el-table-header-bg-color: transparent;
  --el-table-row-hover-bg-color: #fafafa;
}
.oa-el-table :deep(th.el-table__cell) {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--oa-text-3);
  font-weight: 700;
  border-bottom: 1px solid var(--oa-border) !important;
  padding: 10px 8px 12px;
}
.oa-el-table :deep(td.el-table__cell) {
  border-bottom: 1px solid var(--oa-border);
  padding: 12px 8px;
}

/* File cell */
.oa-file-cell {
  display: flex;
  align-items: center;
  gap: 12px;
}
.oa-file-icon {
  width: 40px;
  height: 40px;
  border-radius: 10px;
  background: var(--oa-surface);
  border: 1px solid var(--oa-border);
  display: grid;
  place-content: center;
  color: var(--oa-text);
}
.oa-file-name { font-size: 14px; font-weight: 600; }
.oa-file-meta { font-size: 12px; color: var(--oa-text-2); }

/* Status pills */
.oa-status-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 5px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.02em;
}
.oa-status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: currentColor;
}

/* Softer OpenAI-like pastels */
.pill-green { background: #e9f9f3; color: #0f7a5a; }
.pill-blue  { background: #eef3ff; color: #2a5bd7; }
.pill-gray  { background: #f1f2f4; color: #5f6368; }
.pill-red   { background: #fdecec; color: #b42318; }

/* Progress */
.oa-progress {
  display: flex;
  align-items: center;
  gap: 10px;
}
.oa-progress-bar { flex: 1; }
.oa-progress-text {
  font-size: 12px;
  color: var(--oa-text-2);
  width: 34px;
  text-align: right;
  font-variant-numeric: tabular-nums;
}

/* Date + download */
.oa-date {
  font-size: 13px;
  color: var(--oa-text-2);
  font-variant-numeric: tabular-nums;
}
.oa-download-link { text-decoration: none; }
.oa-download { color: var(--oa-text-2) !important; }
.oa-download:hover { color: var(--oa-green) !important; }

@keyframes rotating {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* Responsive */
@media (max-width: 900px) {
  .oa-stats { grid-template-columns: 1fr; }
}
</style>