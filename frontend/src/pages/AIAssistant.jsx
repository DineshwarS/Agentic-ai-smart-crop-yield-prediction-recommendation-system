import React, { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import { Bot, User, Send, Sparkles, AlertCircle, Wrench, CornerDownLeft } from 'lucide-react'
import Header from '../components/Header'
import { agentChat } from '../api'

const QUICK_PROMPTS = [
  "What is the ideal soil pH and N, P, K required for Cotton?",
  "Recommend a better crop for Punjab based on regional soil & climate.",
  "What soil pH and NPK nutrient levels are needed for Rice?",
  "Recommend the best crop for Maharashtra with yield forecast.",
  "I want to grow Rice in Punjab. Is there a better crop for this location?",
]

export default function AIAssistant() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 
`### Welcome to the Agentic Agricultural Decision Assistant! 🌾

I am an intelligent orchestration agent connected directly to our **Trained Machine Learning Models** (XGBoost Yield Engine & Random Forest Recommendation Classifier) and the **Live Open-Meteo Weather Service**.

**I can autonomously execute tools to:**
* 🌡️ Retrieve live weather & annual climate estimates for your coordinates
* 🌱 Recommend optimal crops based on N-P-K-pH soil chemistry
* 📈 Forecast crop yield in tonnes/ha without data leakage
* ⚖️ Compare multiple candidate crops side-by-side

How can I assist your agricultural planning today?`,
    },
  ])

  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [lastTools, setLastTools] = useState([])
  const messagesEndRef = useRef(null)
  const textareaRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, loading])

  const handleSend = async (textToSend) => {
    const query = (textToSend || input).trim()
    if (!query || loading) return

    const newMsgs = [...messages, { role: 'user', content: query }]
    setMessages(newMsgs)
    setInput('')
    setLoading(true)

    try {
      const res = await agentChat({
        message: query,
        session_id: sessionId,
      })

      if (res.session_id) setSessionId(res.session_id)
      if (res.tool_calls_made && res.tool_calls_made.length > 0) {
        setLastTools(res.tool_calls_made)
      }

      setMessages([...newMsgs, { role: 'assistant', content: res.reply, tools: res.tool_calls_made }])
    } catch (err) {
      setMessages([
        ...newMsgs,
        {
          role: 'assistant',
          content: `⚠️ Failed to get agent response: ${err.message}. If the AI key is not set, you can still use the direct Yield Prediction and Crop Recommendation form pages.`,
        },
      ])
    } finally {
      setLoading(false)
      if (textareaRef.current) {
        textareaRef.current.focus()
      }
    }
  }

  return (
    <div className="main-content">
      <Header
        title="Agentic AI Agricultural Assistant"
        subtitle="Conversational agent with autonomous tool execution: weather lookup, ML yield regression, and crop ranking"
      />

      <div className="page-content">
        <div className="card chat-container">
          {/* Quick Prompts */}
          <div className="quick-prompts-bar mb-3">
            <span className="text-xs text-muted font-medium mr-1">Suggested questions:</span>
            {QUICK_PROMPTS.map((prompt, idx) => (
              <button
                key={idx}
                className="quick-prompt-chip"
                onClick={() => handleSend(prompt)}
                disabled={loading}
              >
                {prompt}
              </button>
            ))}
          </div>

          {/* Messages list */}
          <div className="chat-messages">
            {messages.map((m, idx) => (
              <div key={idx} className={`message ${m.role}`}>
                <div className="message-avatar">
                  {m.role === 'assistant' ? <Bot size={18} /> : <User size={18} />}
                </div>
                <div className="message-bubble">
                  {m.tools && m.tools.length > 0 && (
                    <div className="flex gap-1 mb-2">
                      <span className="badge badge-amber text-xs flex gap-1">
                        <Wrench size={12} />
                        <span>Autonomous Tools: {m.tools.join(', ')}</span>
                      </span>
                    </div>
                  )}
                  <ReactMarkdown>{m.content}</ReactMarkdown>
                </div>
              </div>
            ))}

            {loading && (
              <div className="message assistant">
                <div className="message-avatar">
                  <Bot size={18} />
                </div>
                <div className="message-bubble flex gap-2" style={{ alignItems: 'center' }}>
                  <div className="spinner"></div>
                  <span className="text-sm text-muted">
                    Agent reasoning & calling ML / weather tools...
                  </span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Chat Input Bar */}
          <div className="chat-input-wrapper">
            <form
              className="chat-input-form"
              onSubmit={e => {
                e.preventDefault()
                handleSend()
              }}
            >
              <div className="chat-textarea-container">
                <textarea
                  ref={textareaRef}
                  rows={2}
                  placeholder="Ask anything about crops, soil suitability, expected yield, or weather..."
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      handleSend()
                    }
                  }}
                  disabled={loading}
                  className="chat-textarea"
                />
                <div className="chat-input-helper">
                  <span>Press <strong>Enter ↵</strong> to send, <strong>Shift + Enter</strong> for new line</span>
                </div>
              </div>

              <button
                type="submit"
                className="chat-send-btn"
                disabled={loading || !input.trim()}
              >
                <Send size={18} />
                <span>Send</span>
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  )
}
