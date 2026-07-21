import { Component, type ReactNode } from "react"
import { useLocale } from "./i18n"

function ErrorFallback({ error, onRetry }: { error: Error; onRetry: () => void }) {
  const { t } = useLocale()
  return (
    <div style={{ padding: "2rem", fontFamily: "ui-monospace, monospace", color: "#e5e7eb", background: "#0b0f10", minHeight: "100vh" }}>
      <h2 style={{ color: "#4ed6a5" }}>{t("error.title")}</h2>
      <p style={{ color: "#9ca3af" }}>{t("error.hint")}</p>
      <pre style={{ whiteSpace: "pre-wrap", color: "#f87171" }}>{String(error)}</pre>
      <button onClick={onRetry} style={{ marginTop: "1rem", padding: "0.5rem 1rem", background: "#1f2937", color: "#e5e7eb", border: "1px solid #374151", borderRadius: 8, cursor: "pointer" }}>{t("error.retry")}</button>
    </div>
  )
}

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
    return <ErrorFallback error={this.state.error} onRetry={() => this.setState({ error: null, stack: "" })} />
  }
}
