import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
})

// Request interceptor: attach JWT token if available
api.interceptors.request.use(
  config => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  error => Promise.reject(error)
)

// Response interceptor: handle errors & 401s
api.interceptors.response.use(
  response => response.data,
  error => {
    if (error.response?.status === 401) {
      // If token expired on protected request
      const isAuthUrl = error.config?.url?.includes('/auth/login') || error.config?.url?.includes('/auth/register')
      if (!isAuthUrl) {
        localStorage.removeItem('token')
        localStorage.removeItem('user')
      }
    }
    const msg = error.response?.data?.detail || error.message || 'Unknown error'
    return Promise.reject(new Error(msg))
  }
)

export default api

// ── Auth API Functions ──────────────────────────────

export const registerUser = (data) => api.post('/auth/register', data)
export const loginUser = (data) => api.post('/auth/login', data)
export const getMe = () => api.get('/auth/me')

// ── ML & Weather API Functions ─────────────────────

export const healthCheck = () => api.get('/health')

export const getWeather = (lat, lon, location = '') =>
  api.get('/weather', { params: { lat, lon, location } })

export const predictYield = (data) => api.post('/predict-yield', data)

export const recommendCrop = (data) => api.post('/recommend-crop', data)

export const getCrops = () => api.get('/crops')
export const getStates = () => api.get('/states')
export const getSeasons = () => api.get('/seasons')
export const getStateSoil = (state) => api.get(`/state-soil/${encodeURIComponent(state)}`)
export const getRecommendableCrops = () => api.get('/recommendable-crops')

export const agentChat = (data) => api.post('/agent/chat', data)

export const getPredictions = (limit = 50) =>
  api.get('/predictions', { params: { limit } })

export const getPredictionById = (id) => api.get(`/predictions/${id}`)

export const getModelPerformance = () => api.get('/model-performance')
