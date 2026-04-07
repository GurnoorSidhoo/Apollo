import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

// ─── Types ─────────────────────────────────────────────────────────────────────

type AgentState = 'idle' | 'listening' | 'thinking' | 'executing'

interface Message {
  id: string
  role: 'user' | 'assistant'
  text: string
  timestamp: string
}

interface ActionItem {
  id: string
  label: string
  status: 'running' | 'done' | 'failed' | 'queued'
  detail?: string
  timestamp: string
}

interface WSEvent {
  type: string
  time?: string
  [key: string]: unknown
}

// ─── WebSocket Hook ────────────────────────────────────────────────────────────

function useBiggieBridge() {
  const [connected, setConnected] = useState(false)
  const [agentState, setAgentState] = useState<AgentState>('idle')
  const [transcript, setTranscript] = useState('')
  const [pttHeld, setPttHeld] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [actions, setActions] = useState<ActionItem[]>([])
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectRef = useRef<ReturnType<typeof setTimeout>>(undefined)

  const handleEvent = useCallback((evt: WSEvent) => {
    switch (evt.type) {
      case 'snapshot':
        // Initial state from server
        if (evt.state && typeof evt.state === 'object') {
          const s = evt.state as { agent_state?: string; transcript?: string; ptt_held?: boolean }
          setAgentState((s.agent_state as AgentState) || 'idle')
          setTranscript(s.transcript || '')
          setPttHeld(s.ptt_held || false)
        }
        if (Array.isArray(evt.messages)) setMessages(evt.messages as Message[])
        if (Array.isArray(evt.actions)) setActions(evt.actions as ActionItem[])
        break

      case 'state_change':
        setAgentState((evt.state as AgentState) || 'idle')
        break

      case 'transcript':
        setTranscript((evt.text as string) || '')
        break

      case 'ptt':
        setPttHeld(evt.held as boolean || false)
        break

      case 'message': {
        const msg: Message = {
          id: String(Date.now()) + Math.random(),
          role: (evt.role as 'user' | 'assistant') || 'assistant',
          text: (evt.text as string) || '',
          timestamp: (evt.time as string) || new Date().toISOString(),
        }
        setMessages(prev => [...prev.slice(-100), msg])
        break
      }

      case 'action': {
        const action: ActionItem = {
          id: (evt.id as string) || String(Date.now()) + Math.random(),
          label: (evt.label as string) || '',
          status: (evt.status as ActionItem['status']) || 'done',
          detail: (evt.detail as string) || '',
          timestamp: (evt.time as string) || new Date().toISOString(),
        }
        setActions(prev => [...prev.slice(-50), action])
        break
      }

      case 'action_update': {
        const aid = evt.id as string
        setActions(prev => prev.map(a =>
          a.id === aid
            ? { ...a, ...(evt.status ? { status: evt.status as ActionItem['status'] } : {}), ...(evt.detail ? { detail: evt.detail as string } : {}) }
            : a
        ))
        break
      }
    }
  }, [])

  const sendCommand = useCallback((text: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'command', text }))
    }
  }, [])

  useEffect(() => {
    let disposed = false

    const connect = () => {
      if (disposed) return
      if (wsRef.current?.readyState === WebSocket.OPEN || wsRef.current?.readyState === WebSocket.CONNECTING) return

      const ws = new WebSocket('ws://127.0.0.1:8765')
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
        if (reconnectRef.current) clearTimeout(reconnectRef.current)
      }

      ws.onclose = () => {
        setConnected(false)
        if (wsRef.current === ws) wsRef.current = null
        if (!disposed) reconnectRef.current = setTimeout(connect, 2000)
      }

      ws.onerror = () => {
        ws.close()
      }

      ws.onmessage = (evt) => {
        try {
          const data: WSEvent = JSON.parse(evt.data)
          handleEvent(data)
        } catch { /* ignore malformed */ }
      }
    }

    connect()
    return () => {
      disposed = true
      if (reconnectRef.current) clearTimeout(reconnectRef.current)
      wsRef.current?.close()
      wsRef.current = null
    }
  }, [handleEvent])

  return { connected, agentState, transcript, pttHeld, messages, actions, sendCommand }
}

// ─── Helpers ───────────────────────────────────────────────────────────────────

function formatTime(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true })
  } catch {
    return ''
  }
}

// ─── Canvas Orb (GPU-accelerated) ──────────────────────────────────────────────

function OrbCanvas({ state }: { state: AgentState }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const frameRef = useRef(0)

  const palette = useMemo(() => ({
    idle:      { r: 56, g: 189, b: 248, pulse: 0.004, intensity: 0.35 },
    listening: { r: 56, g: 220, b: 220, pulse: 0.025, intensity: 0.7 },
    thinking:  { r: 120, g: 100, b: 255, pulse: 0.015, intensity: 0.55 },
    executing: { r: 45, g: 212, b: 191, pulse: 0.02, intensity: 0.6 },
  }), [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!
    const size = 400
    canvas.width = size
    canvas.height = size
    const cx = size / 2, cy = size / 2

    let t = 0
    const draw = () => {
      const c = palette[state]
      t += c.pulse
      ctx.clearRect(0, 0, size, size)

      // Outer ambient glow
      const outerGlow = ctx.createRadialGradient(cx, cy, 60, cx, cy, 190)
      outerGlow.addColorStop(0, `rgba(${c.r},${c.g},${c.b},${0.08 + Math.sin(t) * 0.03 * c.intensity})`)
      outerGlow.addColorStop(0.5, `rgba(${c.r},${c.g},${c.b},${0.03 + Math.sin(t * 1.3) * 0.02 * c.intensity})`)
      outerGlow.addColorStop(1, 'rgba(0,0,0,0)')
      ctx.fillStyle = outerGlow
      ctx.fillRect(0, 0, size, size)

      // Pulse rings
      if (state === 'listening' || state === 'executing') {
        for (let ring = 0; ring < 3; ring++) {
          const phase = (t * 0.8 + ring * 2.1) % (Math.PI * 2)
          const radius = 70 + Math.sin(phase) * 30 + ring * 15
          const alpha = Math.max(0, 0.15 - ring * 0.04) * (0.5 + Math.sin(phase) * 0.5)
          ctx.beginPath()
          ctx.arc(cx, cy, radius, 0, Math.PI * 2)
          ctx.strokeStyle = `rgba(${c.r},${c.g},${c.b},${alpha})`
          ctx.lineWidth = 1.5
          ctx.stroke()
        }
      }

      // Main orb body
      const orbRadius = 58 + Math.sin(t * 1.2) * 3 * c.intensity
      const orbGrad = ctx.createRadialGradient(cx - 15, cy - 15, 10, cx, cy, orbRadius)
      orbGrad.addColorStop(0, `rgba(${Math.min(255, c.r + 60)},${Math.min(255, c.g + 40)},${Math.min(255, c.b + 40)},${0.5 * c.intensity})`)
      orbGrad.addColorStop(0.4, `rgba(${c.r},${c.g},${c.b},${0.35 * c.intensity})`)
      orbGrad.addColorStop(0.7, `rgba(${c.r},${c.g},${c.b},${0.15 * c.intensity})`)
      orbGrad.addColorStop(1, 'rgba(10,11,16,0.8)')
      ctx.beginPath()
      ctx.arc(cx, cy, orbRadius, 0, Math.PI * 2)
      ctx.fillStyle = orbGrad
      ctx.fill()

      // Thin bright edge
      ctx.beginPath()
      ctx.arc(cx, cy, orbRadius, 0, Math.PI * 2)
      ctx.strokeStyle = `rgba(${c.r},${c.g},${c.b},${0.2 + Math.sin(t) * 0.1})`
      ctx.lineWidth = 1
      ctx.stroke()

      // Inner core glow
      const coreRadius = 18 + Math.sin(t * 2) * 4 * c.intensity
      const coreGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreRadius)
      coreGrad.addColorStop(0, `rgba(${Math.min(255, c.r + 100)},${Math.min(255, c.g + 80)},${Math.min(255, c.b + 80)},${0.6 * c.intensity})`)
      coreGrad.addColorStop(0.5, `rgba(${c.r},${c.g},${c.b},${0.2 * c.intensity})`)
      coreGrad.addColorStop(1, 'rgba(0,0,0,0)')
      ctx.beginPath()
      ctx.arc(cx, cy, coreRadius, 0, Math.PI * 2)
      ctx.fillStyle = coreGrad
      ctx.fill()

      // Thinking: orbital arcs
      if (state === 'thinking') {
        const angle = t * 2
        ctx.beginPath()
        ctx.arc(cx, cy, orbRadius + 12, angle, angle + Math.PI * 0.7)
        ctx.strokeStyle = `rgba(${c.r},${c.g},${c.b},0.4)`
        ctx.lineWidth = 2
        ctx.lineCap = 'round'
        ctx.stroke()
        ctx.lineCap = 'butt'
        ctx.beginPath()
        ctx.arc(cx, cy, orbRadius + 22, -angle * 0.7, -angle * 0.7 + Math.PI * 0.5)
        ctx.strokeStyle = `rgba(${c.r},${c.g},${c.b},0.2)`
        ctx.lineWidth = 1.5
        ctx.stroke()
      }

      // Executing: progress sweep
      if (state === 'executing') {
        const sweep = (t * 0.5) % (Math.PI * 2)
        ctx.beginPath()
        ctx.arc(cx, cy, orbRadius + 8, sweep, sweep + Math.PI * 1.2)
        ctx.strokeStyle = `rgba(${c.r},${c.g},${c.b},0.35)`
        ctx.lineWidth = 2.5
        ctx.lineCap = 'round'
        ctx.stroke()
        ctx.lineCap = 'butt'
      }

      frameRef.current = requestAnimationFrame(draw)
    }

    draw()
    return () => cancelAnimationFrame(frameRef.current)
  }, [state, palette])

  return <canvas ref={canvasRef} className="w-[280px] h-[280px]" />
}

// ─── Waveform Visualizer ───────────────────────────────────────────────────────

function Waveform() {
  const [bars] = useState(() =>
    Array.from({ length: 32 }).map((_, i) => ({
      delay: i * 0.035,
      duration: 0.35 + Math.random() * 0.5,
      maxH: 6 + Math.random() * 18,
    }))
  )

  return (
    <div className="flex items-center justify-center gap-[2.5px] h-7">
      {bars.map((b, i) => (
        <div
          key={i}
          className="w-[2px] rounded-full bg-accent-cyan/60"
          style={{
            animation: `waveform ${b.duration}s ease-in-out infinite`,
            animationDelay: `${b.delay}s`,
            ['--wave-height' as string]: `${b.maxH}px`,
          }}
        />
      ))}
    </div>
  )
}

// ─── AI Orb Composite ──────────────────────────────────────────────────────────

function AIOrb({ state }: { state: AgentState }) {
  const labels: Record<AgentState, string> = {
    idle: 'READY',
    listening: 'LISTENING',
    thinking: 'THINKING',
    executing: 'EXECUTING',
  }
  const dotColor: Record<AgentState, string> = {
    idle: 'bg-accent-cyan/50',
    listening: 'bg-accent-cyan',
    thinking: 'bg-accent-blue',
    executing: 'bg-accent-teal',
  }

  return (
    <div className="relative flex flex-col items-center gap-2 select-none">
      <OrbCanvas state={state} />
      <AnimatePresence>
        {state === 'listening' && (
          <motion.div
            initial={{ opacity: 0, scaleY: 0 }}
            animate={{ opacity: 1, scaleY: 1 }}
            exit={{ opacity: 0, scaleY: 0 }}
            transition={{ duration: 0.3 }}
            className="-mt-2"
          >
            <Waveform />
          </motion.div>
        )}
      </AnimatePresence>
      <motion.div
        key={state}
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="flex items-center gap-2 mt-1"
      >
        <div className={`w-1.5 h-1.5 rounded-full ${dotColor[state]} ${state !== 'idle' ? 'animate-pulse' : ''}`} />
        <span className="text-[11px] font-medium tracking-[0.22em] text-text-secondary/80">
          {labels[state]}
        </span>
      </motion.div>
    </div>
  )
}

// ─── Conversation Panel ────────────────────────────────────────────────────────

function ConversationPanel({ messages, transcript, agentState }: {
  messages: Message[]
  transcript: string
  agentState: AgentState
}) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, transcript, agentState])

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-border-subtle">
        <h2 className="text-[11px] font-semibold tracking-[0.16em] text-text-muted/70 uppercase">Conversation</h2>
        <span className="text-[10px] text-text-muted/50 tabular-nums">{messages.length}</span>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3.5">
        {messages.length === 0 && agentState === 'idle' && (
          <div className="flex items-center justify-center h-full">
            <p className="text-[12px] text-text-muted/30 text-center leading-relaxed">
              Say &quot;Biggie&quot; followed by a command.<br />
              Or type below.
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <motion.div
            key={msg.id}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i < 3 ? i * 0.04 : 0, duration: 0.35 }}
            className={`flex flex-col gap-1 ${msg.role === 'user' ? 'items-end' : 'items-start'}`}
          >
            <span className="text-[9.5px] text-text-muted/50 tracking-wider uppercase px-1">
              {msg.role === 'user' ? 'You' : 'Biggie'} {msg.timestamp ? `\u00b7 ${formatTime(msg.timestamp)}` : ''}
            </span>
            <div className={`max-w-[92%] px-3.5 py-2.5 text-[13px] leading-[1.55] ${
              msg.role === 'user'
                ? 'bg-accent-cyan/[0.07] text-text-primary rounded-2xl rounded-br-lg border border-accent-cyan/[0.08]'
                : 'bg-bg-tertiary/70 text-text-secondary rounded-2xl rounded-bl-lg border border-border-subtle/50'
            }`}>
              {msg.text}
            </div>
          </motion.div>
        ))}

        {/* Live transcript while listening */}
        <AnimatePresence>
          {agentState === 'listening' && transcript && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="flex flex-col gap-1 items-end"
            >
              <span className="text-[9.5px] text-accent-cyan/70 tracking-wider uppercase px-1">Hearing&hellip;</span>
              <div className="max-w-[92%] px-3.5 py-2.5 rounded-2xl rounded-br-lg bg-accent-cyan/[0.07] text-text-primary text-[13px] leading-[1.55] border border-accent-cyan/20">
                {transcript}
                <span className="inline-block w-[2px] h-3.5 bg-accent-cyan/80 ml-1 animate-pulse align-middle" />
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Thinking indicator */}
        <AnimatePresence>
          {agentState === 'thinking' && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="flex flex-col gap-1 items-start"
            >
              <span className="text-[9.5px] text-accent-blue/70 tracking-wider uppercase px-1">Processing&hellip;</span>
              <div className="px-4 py-3 rounded-2xl rounded-bl-lg bg-bg-tertiary/70 border border-border-subtle/50">
                <div className="flex gap-1.5">
                  {[0, 1, 2].map(j => (
                    <div key={j} className="w-1.5 h-1.5 rounded-full bg-accent-blue/50 animate-pulse" style={{ animationDelay: `${j * 0.2}s` }} />
                  ))}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <div ref={bottomRef} />
      </div>
    </div>
  )
}

// ─── Command Input ─────────────────────────────────────────────────────────────

function CommandInput({ onSend }: { onSend: (text: string) => void }) {
  const [value, setValue] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = value.trim()
    if (!trimmed) return
    onSend(trimmed)
    setValue('')
  }

  return (
    <form onSubmit={handleSubmit} className="px-4 pb-3 pt-2 border-t border-border-subtle">
      <div className="flex items-center gap-2 bg-white/[0.03] rounded-xl border border-white/[0.06] px-3 py-2">
        <input
          type="text"
          value={value}
          onChange={e => setValue(e.target.value)}
          placeholder="Type a command..."
          className="flex-1 bg-transparent text-[13px] text-text-primary placeholder:text-text-muted/30 outline-none"
        />
        <button
          type="submit"
          className="text-[10px] tracking-wider text-accent-cyan/60 hover:text-accent-cyan transition-colors cursor-pointer bg-transparent border-none"
        >
          SEND
        </button>
      </div>
    </form>
  )
}

// ─── Action Feed ───────────────────────────────────────────────────────────────

function ActionFeed({ actions }: { actions: ActionItem[] }) {
  const statusStyles: Record<string, { dot: string; badge: string; bg: string }> = {
    running: {
      dot: 'bg-accent-cyan animate-pulse',
      badge: 'text-accent-cyan bg-accent-cyan/10 border-accent-cyan/20',
      bg: 'bg-accent-cyan/[0.04] border border-accent-cyan/10',
    },
    done: {
      dot: 'bg-accent-teal/60',
      badge: 'text-accent-teal/80 bg-accent-teal/10 border-accent-teal/15',
      bg: '',
    },
    failed: {
      dot: 'bg-accent-rose/60',
      badge: 'text-accent-rose/80 bg-accent-rose/10 border-accent-rose/15',
      bg: '',
    },
    queued: {
      dot: 'bg-text-muted/40',
      badge: 'text-text-muted/60 bg-white/[0.03] border-white/[0.06]',
      bg: '',
    },
  }

  // Show most recent first
  const sorted = [...actions].reverse()

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-5 py-3.5 border-b border-border-subtle">
        <h2 className="text-[11px] font-semibold tracking-[0.16em] text-text-muted/70 uppercase">Activity</h2>
        <div className="flex items-center gap-1.5">
          <div className={`w-1.5 h-1.5 rounded-full ${actions.some(a => a.status === 'running') ? 'bg-accent-cyan animate-pulse' : 'bg-text-muted/30'}`} />
          <span className="text-[10px] text-text-muted/50 tabular-nums">
            {actions.filter(a => a.status === 'running').length} active
          </span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-1">
        {sorted.length === 0 && (
          <p className="text-[11px] text-text-muted/25 text-center py-8">No activity yet</p>
        )}
        {sorted.map((action, i) => {
          const s = statusStyles[action.status] || statusStyles.done
          return (
            <motion.div
              key={action.id + action.timestamp}
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i < 5 ? i * 0.04 : 0, duration: 0.25 }}
              className={`flex items-start gap-3 px-3.5 py-3 rounded-xl transition-colors ${s.bg}`}
            >
              <div className={`w-2 h-2 rounded-full mt-1.5 shrink-0 ${s.dot}`} />
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[12.5px] text-text-primary/90 truncate leading-snug">{action.label}</span>
                  <span className={`text-[9.5px] font-medium uppercase tracking-wider px-2 py-0.5 rounded-full border shrink-0 ${s.badge}`}>
                    {action.status}
                  </span>
                </div>
                {action.detail && (
                  <span className="text-[11px] text-text-muted/60 leading-snug">{action.detail}</span>
                )}
              </div>
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}

// ─── Status Bar ────────────────────────────────────────────────────────────────

function StatusBar({ connected, agentState, pttHeld }: { connected: boolean; agentState: AgentState; pttHeld: boolean }) {
  const [time, setTime] = useState(new Date())

  useEffect(() => {
    const iv = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(iv)
  }, [])

  return (
    <div className="flex items-center justify-between px-6 py-2 text-[10.5px] text-text-muted/40 font-mono tracking-wide">
      <div className="flex items-center gap-4">
        <span className="tabular-nums">
          {time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })}
        </span>
        <span className="opacity-30">&middot;</span>
        <span className={connected ? 'text-accent-teal/60' : 'text-accent-rose/60'}>
          {connected ? '● Connected' : '○ Disconnected'}
        </span>
      </div>
      <div className="flex items-center gap-4">
        <span className={pttHeld ? 'text-accent-cyan/70' : ''}>
          {pttHeld ? 'PTT active' : 'PTT ready'}
        </span>
        <span className="opacity-30">&middot;</span>
        <span>Apollo Engine</span>
        <span className="opacity-30">&middot;</span>
        <span className="flex items-center gap-1.5">
          <span className={`w-1.5 h-1.5 rounded-full ${
            agentState !== 'idle' ? 'bg-accent-cyan/70 animate-pulse' : 'bg-accent-cyan/30'
          }`} />
          Biggie
        </span>
      </div>
    </div>
  )
}

// ─── Background Ambient ────────────────────────────────────────────────────────

function AmbientBackground({ state }: { state: AgentState }) {
  const color = state === 'listening' ? '56,220,220' : state === 'thinking' ? '120,100,255' : state === 'executing' ? '45,212,191' : '56,189,248'
  const intensity = state === 'idle' ? 0.06 : 0.12

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      <div
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[900px] h-[700px] rounded-full blur-[120px] transition-all duration-1000"
        style={{ background: `radial-gradient(ellipse, rgba(${color},${intensity}), transparent 70%)` }}
      />
      <div className="absolute top-0 left-0 w-full h-48 bg-gradient-to-b from-bg-secondary/50 to-transparent" />
      <div className="absolute bottom-0 left-0 w-full h-32 bg-gradient-to-t from-bg-primary to-transparent" />
      <div className="absolute top-0 left-0 w-48 h-full bg-gradient-to-r from-bg-primary/60 to-transparent" />
      <div className="absolute top-0 right-0 w-48 h-full bg-gradient-to-l from-bg-primary/60 to-transparent" />
    </div>
  )
}

// ─── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const { connected, agentState, transcript, pttHeld, messages, actions, sendCommand } = useBiggieBridge()

  return (
    <div className="relative w-full h-full flex flex-col bg-bg-primary overflow-hidden">
      <AmbientBackground state={agentState} />

      {/* Top Bar */}
      <header className="relative z-10 flex items-center justify-between px-6 h-12 border-b border-white/[0.04] shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2.5">
            <div className={`w-[7px] h-[7px] rounded-full ${
              connected
                ? 'bg-accent-cyan shadow-[0_0_10px_rgba(56,189,248,0.5)]'
                : 'bg-accent-rose/60 shadow-[0_0_10px_rgba(251,113,133,0.3)]'
            }`} />
            <span className="text-[13px] font-semibold tracking-[0.06em] text-text-primary/90">BIGGIE</span>
          </div>
          <div className="h-3 w-px bg-white/[0.06]" />
          <span className="text-[10px] text-text-muted/40 tracking-[0.12em]">APOLLO ENGINE</span>
        </div>
        {!connected && (
          <span className="text-[10px] text-accent-rose/50 tracking-wider animate-pulse">
            Waiting for Apollo backend&hellip;
          </span>
        )}
      </header>

      {/* Main Content */}
      <div className="relative z-10 flex-1 flex min-h-0">
        {/* Left — Conversation + Input */}
        <aside className="w-[360px] border-r border-white/[0.04] flex flex-col shrink-0 bg-bg-primary/40 backdrop-blur-sm">
          <div className="flex-1 min-h-0">
            <ConversationPanel messages={messages} transcript={transcript} agentState={agentState} />
          </div>
          <CommandInput onSend={sendCommand} />
        </aside>

        {/* Center — Orb + PTT hint */}
        <main className="flex-1 flex flex-col items-center justify-center relative min-w-0">
          <AIOrb state={agentState} />
          <div className="mt-6">
            <AnimatePresence mode="wait">
              {pttHeld ? (
                <motion.span
                  key="ptt-active"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="text-[10px] tracking-[0.18em] text-accent-cyan/60 uppercase"
                >
                  Listening &mdash; release to send
                </motion.span>
              ) : agentState === 'idle' && connected ? (
                <motion.span
                  key="ptt-hint"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="text-[10px] tracking-[0.18em] text-text-muted/25 uppercase"
                >
                  Hold F9 to talk
                </motion.span>
              ) : null}
            </AnimatePresence>
          </div>
        </main>

        {/* Right — Activity */}
        <aside className="w-[300px] border-l border-white/[0.04] flex flex-col shrink-0 bg-bg-primary/40 backdrop-blur-sm overflow-hidden">
          <ActionFeed actions={actions} />
        </aside>
      </div>

      {/* Status Bar */}
      <footer className="relative z-10 border-t border-white/[0.04] shrink-0">
        <StatusBar connected={connected} agentState={agentState} pttHeld={pttHeld} />
      </footer>
    </div>
  )
}
