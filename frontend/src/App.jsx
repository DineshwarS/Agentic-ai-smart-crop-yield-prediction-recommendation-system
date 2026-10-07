import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './context/AuthContext'
import Sidebar from './components/Sidebar'
import Dashboard from './pages/Dashboard'
import YieldPrediction from './pages/YieldPrediction'
import CropRecommendation from './pages/CropRecommendation'
import AIAssistant from './pages/AIAssistant'
import PredictionHistory from './pages/PredictionHistory'
import ModelPerformance from './pages/ModelPerformance'
import Login from './pages/Login'
import Register from './pages/Register'

function ProtectedLayout() {
  const { isAuthenticated, loading } = useAuth()

  if (loading) {
    return (
      <div className="auth-page">
        <div className="spinner" style={{ width: 40, height: 40 }} />
        <p className="text-muted mt-4">Verifying session...</p>
      </div>
    )
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />
  }

  return (
    <div className="app-layout">
      <Sidebar />
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/predict-yield" element={<YieldPrediction />} />
        <Route path="/recommend-crop" element={<CropRecommendation />} />
        <Route path="/ai-assistant" element={<AIAssistant />} />
        <Route path="/history" element={<PredictionHistory />} />
        <Route path="/model-performance" element={<ModelPerformance />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  )
}

function PublicRoute({ children }) {
  const { isAuthenticated, loading } = useAuth()

  if (loading) {
    return null
  }

  if (isAuthenticated) {
    return <Navigate to="/" replace />
  }

  return children
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route
            path="/login"
            element={
              <PublicRoute>
                <Login />
              </PublicRoute>
            }
          />
          <Route
            path="/register"
            element={
              <PublicRoute>
                <Register />
              </PublicRoute>
            }
          />
          <Route path="/*" element={<ProtectedLayout />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  )
}
