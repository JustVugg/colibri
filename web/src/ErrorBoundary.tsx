import { Component, type ReactNode } from "react"
interface State { error: Error | null; stack: string }
export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null, stack: "" }
  static getDerivedStateFromError(error: Error): Partial<State> { return { error } }
  componentDidCatch(error: Error, info: { componentStack?: string }) {
    console.error("[colibrì] render crash:", error, info.componentStack)
    this.setState({ stack: info.componentStack ?? "" })
  }
  render() {
    if (!this.state.error) return this.props.children
    return <div style={{ padding: "2rem", fontFamily: "ui-monospace, monospace", color: "#e5e7eb", background: "#0b0f10", minHeight: "100vh" }}>
      <h2 style={{ color: "#4ed6a5" }}>colibrì UI hit an error</h2>
      <p style={{ color: "#9ca3af" }}>The engine is unaffected. Try refreshing.</p>
      <pre style={{ whiteSpace: "pre-wrap", color: "#f87171" }}>{String(this.state.error)}</pre>
      <button onClick={() => this.setState({ error: null, stack: "" })} style={{ marginTop: "1rem", padding: "0.5rem 1rem", background: "#1f2937", color: "#e5e7eb", border: "1px solid #374151", borderRadius: 8, cursor: "pointer" }}>Retry</button>
    </div>
  }
}
