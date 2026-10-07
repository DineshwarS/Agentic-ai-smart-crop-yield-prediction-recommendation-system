import React from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { 
  LayoutDashboard, 
  Sprout, 
  Sparkles, 
  Bot, 
  History, 
  BarChart3, 
  LogOut,
  User
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'

export default function Sidebar() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const navItems = [
    { to: '/', label: 'Dashboard', icon: LayoutDashboard },
    { to: '/predict-yield', label: 'Yield Prediction', icon: Sprout },
    { to: '/recommend-crop', label: 'Crop Recommendation', icon: Sparkles },
    { to: '/ai-assistant', label: 'AI Agricultural Agent', icon: Bot },
    { to: '/history', label: 'Prediction History', icon: History },
    { to: '/model-performance', label: 'Model Performance', icon: BarChart3 },
  ]

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-icon">🌾</div>
        <div>
          <div className="logo-text">CropAI</div>
          <div className="logo-sub">AGENTIC DECISION SYSTEM</div>
        </div>
      </div>

      <div className="sidebar-nav">
        <div className="nav-section-label">Core Modules</div>
        {navItems.map((item) => {
          const Icon = item.icon
          return (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            >
              <Icon size={18} className="nav-icon" />
              <span>{item.label}</span>
            </NavLink>
          )
        })}
      </div>

      <div className="sidebar-footer">
        {user && (
          <div className="user-profile-widget mb-3">
            <div className="user-avatar-circle">
              <User size={16} />
            </div>
            <div className="user-info-text">
              <div className="user-name">{user.name || 'Farmer / User'}</div>
              <div className="user-email">{user.email}</div>
            </div>
            <button
              onClick={handleLogout}
              className="btn-logout"
              title="Sign Out"
            >
              <LogOut size={16} />
            </button>
          </div>
        )}

        <div className="status-badge">
          <span className="status-dot"></span>
          <span>ML & Agent Services Active</span>
        </div>
      </div>
    </aside>
  )
}
