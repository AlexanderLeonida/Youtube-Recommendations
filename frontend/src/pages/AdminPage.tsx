import React, { useEffect, useState, useMemo } from "react";
import { api } from "../services/api";

interface VideoData {
  id: number;
  title: string;
  channel_name: string;
  view_count: string;
  duration: string;
  extracted_at: string;
}

interface EvalMetrics {
  recall_at_k: Record<string, number>;
  ndcg_at_k: Record<string, number>;
  hit_rate_at_k: Record<string, number>;
  mrr: number;
  coverage: number;
  catalog_size: number;
  num_eval_sessions: number;
  latency: {
    p50_ms: number;
    p95_ms: number;
    p99_ms: number;
    mean_ms: number;
    num_runs: number;
  };
  loss_history: number[];
  ctr_stats: {
    overall_ctr: number;
    total_impressions: number;
    total_clicks: number;
    unique_videos: number;
  };
}

const ADMIN_PASSWORD = "password";

// ── SVG Chart Components ──────────────────────────────────────────────────

function BarChart({
  data,
  title,
  color = "#1976D2",
  height = 200,
  formatValue = (v: number) => (v * 100).toFixed(1) + "%",
  target,
}: {
  data: { label: string; value: number }[];
  title: string;
  color?: string;
  height?: number;
  formatValue?: (v: number) => string;
  target?: { value: number; label: string };
}) {
  if (data.length === 0) return null;
  const maxVal = Math.max(...data.map((d) => d.value), target?.value ?? 0) * 1.15 || 1;
  const barWidth = Math.min(60, (400 - data.length * 8) / data.length);
  const chartWidth = data.length * (barWidth + 16) + 60;
  const chartHeight = height;
  const plotH = chartHeight - 40;

  return (
    <div style={styles.chartCard}>
      <h3 style={styles.chartTitle}>{title}</h3>
      <svg width={chartWidth} height={chartHeight} style={{ overflow: "visible" }}>
        {/* Y-axis gridlines */}
        {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
          const y = plotH - frac * plotH;
          return (
            <g key={frac}>
              <line x1={40} y1={y} x2={chartWidth} y2={y} stroke="#eee" strokeWidth={1} />
              <text x={36} y={y + 4} textAnchor="end" fontSize={10} fill="#999">
                {formatValue(frac * maxVal)}
              </text>
            </g>
          );
        })}
        {/* Target line */}
        {target && (
          <g>
            <line
              x1={40}
              y1={plotH - (target.value / maxVal) * plotH}
              x2={chartWidth}
              y2={plotH - (target.value / maxVal) * plotH}
              stroke="#e53935"
              strokeWidth={1.5}
              strokeDasharray="6,3"
            />
            <text
              x={chartWidth + 2}
              y={plotH - (target.value / maxVal) * plotH + 4}
              fontSize={10}
              fill="#e53935"
            >
              {target.label}
            </text>
          </g>
        )}
        {/* Bars */}
        {data.map((d, i) => {
          const barH = (d.value / maxVal) * plotH;
          const x = 48 + i * (barWidth + 16);
          return (
            <g key={d.label}>
              <rect
                x={x}
                y={plotH - barH}
                width={barWidth}
                height={barH}
                fill={color}
                rx={3}
                opacity={0.85}
              />
              <text
                x={x + barWidth / 2}
                y={plotH - barH - 6}
                textAnchor="middle"
                fontSize={11}
                fontWeight={600}
                fill="#333"
              >
                {formatValue(d.value)}
              </text>
              <text
                x={x + barWidth / 2}
                y={plotH + 16}
                textAnchor="middle"
                fontSize={11}
                fill="#666"
              >
                {d.label}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function LineChart({
  data,
  title,
  color = "#1976D2",
  height = 200,
  xLabel = "Epoch",
  yLabel = "Loss",
}: {
  data: number[];
  title: string;
  color?: string;
  height?: number;
  xLabel?: string;
  yLabel?: string;
}) {
  if (data.length === 0) return null;
  const maxVal = Math.max(...data) * 1.1 || 1;
  const minVal = Math.min(...data) * 0.9;
  const range = maxVal - minVal || 1;
  const chartWidth = Math.max(400, data.length * 20 + 80);
  const chartHeight = height;
  const plotH = chartHeight - 40;
  const plotW = chartWidth - 60;

  const points = data.map((v, i) => {
    const x = 50 + (i / Math.max(data.length - 1, 1)) * plotW;
    const y = plotH - ((v - minVal) / range) * plotH;
    return `${x},${y}`;
  });

  return (
    <div style={styles.chartCard}>
      <h3 style={styles.chartTitle}>{title}</h3>
      <svg width={chartWidth} height={chartHeight} style={{ overflow: "visible" }}>
        {/* Y-axis labels */}
        {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
          const y = plotH - frac * plotH;
          const val = minVal + frac * range;
          return (
            <g key={frac}>
              <line x1={48} y1={y} x2={chartWidth - 10} y2={y} stroke="#eee" strokeWidth={1} />
              <text x={44} y={y + 4} textAnchor="end" fontSize={10} fill="#999">
                {val.toFixed(3)}
              </text>
            </g>
          );
        })}
        {/* Line */}
        <polyline
          points={points.join(" ")}
          fill="none"
          stroke={color}
          strokeWidth={2}
          strokeLinejoin="round"
        />
        {/* Dots */}
        {data.map((v, i) => {
          const x = 50 + (i / Math.max(data.length - 1, 1)) * plotW;
          const y = plotH - ((v - minVal) / range) * plotH;
          return <circle key={i} cx={x} cy={y} r={3} fill={color} />;
        })}
        {/* X-axis labels (sparse) */}
        {data.map((_, i) => {
          const step = Math.max(1, Math.floor(data.length / 10));
          if (i % step !== 0 && i !== data.length - 1) return null;
          const x = 50 + (i / Math.max(data.length - 1, 1)) * plotW;
          return (
            <text key={i} x={x} y={plotH + 16} textAnchor="middle" fontSize={10} fill="#666">
              {i + 1}
            </text>
          );
        })}
        {/* Axis labels */}
        <text x={chartWidth / 2} y={plotH + 32} textAnchor="middle" fontSize={11} fill="#888">
          {xLabel}
        </text>
        <text
          x={12}
          y={plotH / 2}
          textAnchor="middle"
          fontSize={11}
          fill="#888"
          transform={`rotate(-90, 12, ${plotH / 2})`}
        >
          {yLabel}
        </text>
      </svg>
    </div>
  );
}

function MetricCard({
  label,
  value,
  subtitle,
  color = "#1976D2",
}: {
  label: string;
  value: string;
  subtitle?: string;
  color?: string;
}) {
  return (
    <div style={styles.metricCard}>
      <div style={{ fontSize: 12, color: "#888", textTransform: "uppercase", letterSpacing: 1 }}>
        {label}
      </div>
      <div style={{ fontSize: 28, fontWeight: 700, color, marginTop: 4 }}>{value}</div>
      {subtitle && <div style={{ fontSize: 12, color: "#aaa", marginTop: 2 }}>{subtitle}</div>}
    </div>
  );
}

function LatencyChart({ latency }: { latency: EvalMetrics["latency"] }) {
  const data = [
    { label: "P50", value: latency.p50_ms },
    { label: "P95", value: latency.p95_ms },
    { label: "P99", value: latency.p99_ms },
    { label: "Mean", value: latency.mean_ms },
  ];
  const maxVal = Math.max(...data.map((d) => d.value), 12) * 1.2;
  const chartWidth = 360;
  const chartHeight = 180;
  const plotH = chartHeight - 40;
  const barWidth = 50;

  return (
    <div style={styles.chartCard}>
      <h3 style={styles.chartTitle}>Inference Latency (ms)</h3>
      <div style={{ fontSize: 12, color: "#888", marginBottom: 8 }}>
        {latency.num_runs} inference runs
      </div>
      <svg width={chartWidth} height={chartHeight} style={{ overflow: "visible" }}>
        {/* Target lines */}
        <line
          x1={40}
          y1={plotH - (8 / maxVal) * plotH}
          x2={chartWidth}
          y2={plotH - (8 / maxVal) * plotH}
          stroke="#4CAF50"
          strokeWidth={1.5}
          strokeDasharray="6,3"
        />
        <text
          x={chartWidth + 2}
          y={plotH - (8 / maxVal) * plotH + 4}
          fontSize={10}
          fill="#4CAF50"
        >
          P50 target (8ms)
        </text>
        <line
          x1={40}
          y1={plotH - (12 / maxVal) * plotH}
          x2={chartWidth}
          y2={plotH - (12 / maxVal) * plotH}
          stroke="#e53935"
          strokeWidth={1.5}
          strokeDasharray="6,3"
        />
        <text
          x={chartWidth + 2}
          y={plotH - (12 / maxVal) * plotH + 4}
          fontSize={10}
          fill="#e53935"
        >
          P99 target (12ms)
        </text>
        {/* Bars */}
        {data.map((d, i) => {
          const barH = (d.value / maxVal) * plotH;
          const x = 48 + i * (barWidth + 20);
          const meetTarget =
            (d.label === "P50" && d.value < 8) ||
            (d.label === "P99" && d.value < 12) ||
            (d.label !== "P50" && d.label !== "P99");
          return (
            <g key={d.label}>
              <rect
                x={x}
                y={plotH - barH}
                width={barWidth}
                height={barH}
                fill={meetTarget ? "#4CAF50" : "#FF9800"}
                rx={3}
                opacity={0.85}
              />
              <text
                x={x + barWidth / 2}
                y={plotH - barH - 6}
                textAnchor="middle"
                fontSize={11}
                fontWeight={600}
                fill="#333"
              >
                {d.value.toFixed(2)}ms
              </text>
              <text
                x={x + barWidth / 2}
                y={plotH + 16}
                textAnchor="middle"
                fontSize={11}
                fill="#666"
              >
                {d.label}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────

export default function AdminPage() {
  const [authenticated, setAuthenticated] = useState(false);
  const [passwordInput, setPasswordInput] = useState("");
  const [loginError, setLoginError] = useState("");
  const [tab, setTab] = useState<"videos" | "metrics">("metrics");

  // Videos tab state
  const [videos, setVideos] = useState<VideoData[]>([]);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  // Metrics tab state
  const [metrics, setMetrics] = useState<EvalMetrics | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [metricsError, setMetricsError] = useState("");

  const loadVideos = () => {
    setLoading(true);
    api.getVideos()
      .then(res => {
        if (res.data.videos) setVideos(res.data.videos);
      })
      .catch(err => console.error("Error loading videos:", err))
      .finally(() => setLoading(false));
  };

  const loadMetrics = () => {
    setMetricsLoading(true);
    setMetricsError("");
    api.evaluateModel()
      .then(res => setMetrics(res.data))
      .catch(err => {
        const detail = err.response?.data?.detail || err.message;
        setMetricsError(detail);
      })
      .finally(() => setMetricsLoading(false));
  };

  useEffect(() => {
    if (authenticated && tab === "videos") loadVideos();
  }, [authenticated, tab]); // loadVideos is stable (no deps)

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    if (passwordInput === ADMIN_PASSWORD) {
      setAuthenticated(true);
      setLoginError("");
    } else {
      setLoginError("Incorrect password.");
    }
  };

  const toggleSelect = (id: number) => {
    setSelected(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const toggleSelectAll = () => {
    if (selected.size === videos.length) {
      setSelected(new Set());
    } else {
      setSelected(new Set(videos.map(v => v.id)));
    }
  };

  const deleteVideo = async (id: number) => {
    try {
      await api.deleteVideo(id);
      setVideos(prev => prev.filter(v => v.id !== id));
      setSelected(prev => { const next = new Set(prev); next.delete(id); return next; });
    } catch (err) {
      console.error("Error deleting video:", err);
      setMessage("Failed to delete video.");
    }
  };

  const deleteSelected = async () => {
    if (selected.size === 0) return;
    const ids = Array.from(selected);
    try {
      await Promise.all(ids.map(id => api.deleteVideo(id)));
      setVideos(prev => prev.filter(v => !selected.has(v.id)));
      setSelected(new Set());
      setMessage(`Deleted ${ids.length} video(s).`);
    } catch (err) {
      console.error("Error deleting videos:", err);
      setMessage("Some deletions failed.");
    }
  };

  // ── Prepare chart data from metrics ──
  const recallData = useMemo(() => {
    if (!metrics) return [];
    return Object.entries(metrics.recall_at_k).map(([key, val]) => ({
      label: key.replace("recall@", "K="),
      value: val,
    }));
  }, [metrics]);

  const ndcgData = useMemo(() => {
    if (!metrics) return [];
    return Object.entries(metrics.ndcg_at_k).map(([key, val]) => ({
      label: key.replace("ndcg@", "K="),
      value: val,
    }));
  }, [metrics]);

  const hitRateData = useMemo(() => {
    if (!metrics) return [];
    return Object.entries(metrics.hit_rate_at_k).map(([key, val]) => ({
      label: key.replace("hit_rate@", "K="),
      value: val,
    }));
  }, [metrics]);

  // ── Login screen ──
  if (!authenticated) {
    return (
      <div style={styles.loginWrap}>
        <div style={styles.loginBox}>
          <h2 style={{ marginBottom: 24 }}>Admin Login</h2>
          <form onSubmit={handleLogin}>
            <input
              type="password"
              placeholder="Password"
              value={passwordInput}
              onChange={e => setPasswordInput(e.target.value)}
              style={styles.input}
              autoFocus
            />
            {loginError && <p style={{ color: "#e53935", marginBottom: 8 }}>{loginError}</p>}
            <button type="submit" style={styles.btnPrimary}>Login</button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div style={{ padding: "24px", maxWidth: "1400px", margin: "0 auto" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
        <h1 style={{ margin: 0 }}>TwinTube Vector — Admin</h1>
        <button onClick={() => setAuthenticated(false)} style={styles.btnSecondary}>Logout</button>
      </div>

      {/* Tabs */}
      <div style={styles.tabBar}>
        <button
          onClick={() => setTab("metrics")}
          style={tab === "metrics" ? styles.tabActive : styles.tab}
        >
          Model Metrics
        </button>
        <button
          onClick={() => setTab("videos")}
          style={tab === "videos" ? styles.tabActive : styles.tab}
        >
          Videos
        </button>
      </div>

      {/* ── Metrics Tab ── */}
      {tab === "metrics" && (
        <div>
          <div style={{ display: "flex", gap: 12, marginBottom: 20, alignItems: "center" }}>
            <button onClick={loadMetrics} style={styles.btnPrimary} disabled={metricsLoading}>
              {metricsLoading ? "Evaluating..." : "Run Evaluation"}
            </button>
            {metricsLoading && (
              <span style={{ color: "#888", fontSize: 13 }}>
                Running leave-one-out evaluation and latency benchmarks...
              </span>
            )}
          </div>

          {metricsError && (
            <div style={styles.errorBox}>{metricsError}</div>
          )}

          {!metrics && !metricsLoading && !metricsError && (
            <div style={{ color: "#888", padding: 40, textAlign: "center" }}>
              Click "Run Evaluation" to compute ranking metrics, latency benchmarks, and training curves.
              <br />
              <span style={{ fontSize: 13 }}>Requires a trained model and browse history data.</span>
            </div>
          )}

          {metrics && (
            <>
              {/* ── Summary Cards ── */}
              <div style={styles.cardRow}>
                <MetricCard
                  label="MRR"
                  value={metrics.mrr.toFixed(4)}
                  subtitle="Mean Reciprocal Rank"
                  color="#1976D2"
                />
                <MetricCard
                  label="Coverage"
                  value={(metrics.coverage * 100).toFixed(1) + "%"}
                  subtitle={`${Math.round(metrics.coverage * metrics.catalog_size)} / ${metrics.catalog_size} videos`}
                  color="#7B1FA2"
                />
                <MetricCard
                  label="Overall CTR"
                  value={
                    metrics.ctr_stats.overall_ctr
                      ? (metrics.ctr_stats.overall_ctr * 100).toFixed(2) + "%"
                      : "N/A"
                  }
                  subtitle={`${metrics.ctr_stats.total_clicks} clicks / ${metrics.ctr_stats.total_impressions} impressions`}
                  color="#00897B"
                />
                <MetricCard
                  label="Eval Sessions"
                  value={String(metrics.num_eval_sessions)}
                  subtitle="Leave-one-out users"
                  color="#E65100"
                />
                <MetricCard
                  label="P50 Latency"
                  value={metrics.latency.p50_ms.toFixed(2) + "ms"}
                  subtitle={metrics.latency.p50_ms < 8 ? "Target: < 8ms" : "Target: < 8ms"}
                  color={metrics.latency.p50_ms < 8 ? "#4CAF50" : "#e53935"}
                />
              </div>

              {/* ── Metric Descriptions ── */}
              <div style={styles.descBox}>
                <strong>Metric definitions (standard RecSys evaluation):</strong>
                <ul style={{ margin: "8px 0 0 0", paddingLeft: 20, lineHeight: 1.8 }}>
                  <li><strong>Recall@K</strong> — Fraction of held-out relevant items found in the top-K recommendations. Higher = better retrieval coverage.</li>
                  <li><strong>NDCG@K</strong> — Normalized Discounted Cumulative Gain. Rewards relevant items ranked higher. 1.0 = perfect ranking.</li>
                  <li><strong>Hit Rate@K</strong> — Fraction of evaluation users where the held-out item appeared anywhere in top-K. A coarser but intuitive measure.</li>
                  <li><strong>MRR</strong> — Mean Reciprocal Rank. Average of 1/rank of the first relevant item. Higher = relevant items appear earlier.</li>
                  <li><strong>Coverage</strong> — Fraction of the catalog that appears in at least one recommendation. Higher = less popularity bias.</li>
                  <li><strong>Latency</strong> — End-to-end inference time percentiles. Targets: P50 &lt; 8ms, P99 &lt; 12ms.</li>
                </ul>
              </div>

              {/* ── Charts Row 1: Recall, NDCG ── */}
              <div style={styles.chartRow}>
                <BarChart
                  data={recallData}
                  title="Recall@K"
                  color="#1976D2"
                />
                <BarChart
                  data={ndcgData}
                  title="NDCG@K"
                  color="#7B1FA2"
                />
              </div>

              {/* ── Charts Row 2: Hit Rate, Latency ── */}
              <div style={styles.chartRow}>
                <BarChart
                  data={hitRateData}
                  title="Hit Rate@K"
                  color="#00897B"
                />
                <LatencyChart latency={metrics.latency} />
              </div>

              {/* ── Training Loss Curve ── */}
              {metrics.loss_history.length > 0 && (
                <div style={styles.chartRow}>
                  <LineChart
                    data={metrics.loss_history}
                    title="Training Loss Curve"
                    color="#E65100"
                    xLabel="Epoch"
                    yLabel="BCE Loss"
                  />
                </div>
              )}

              {/* ── Raw numbers table ── */}
              <div style={{ ...styles.chartCard, marginTop: 24 }}>
                <h3 style={styles.chartTitle}>Detailed Results</h3>
                <table style={styles.metricsTable}>
                  <thead>
                    <tr>
                      <th style={styles.mth}>Metric</th>
                      {Object.keys(metrics.recall_at_k).map((k) => (
                        <th key={k} style={styles.mth}>{k.replace("recall@", "K=")}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td style={styles.mtd}><strong>Recall@K</strong></td>
                      {Object.values(metrics.recall_at_k).map((v, i) => (
                        <td key={i} style={styles.mtd}>{(v * 100).toFixed(2)}%</td>
                      ))}
                    </tr>
                    <tr style={{ backgroundColor: "#fafafa" }}>
                      <td style={styles.mtd}><strong>NDCG@K</strong></td>
                      {Object.values(metrics.ndcg_at_k).map((v, i) => (
                        <td key={i} style={styles.mtd}>{(v * 100).toFixed(2)}%</td>
                      ))}
                    </tr>
                    <tr>
                      <td style={styles.mtd}><strong>Hit Rate@K</strong></td>
                      {Object.values(metrics.hit_rate_at_k).map((v, i) => (
                        <td key={i} style={styles.mtd}>{(v * 100).toFixed(2)}%</td>
                      ))}
                    </tr>
                  </tbody>
                </table>

                <table style={{ ...styles.metricsTable, marginTop: 16 }}>
                  <thead>
                    <tr>
                      <th style={styles.mth}>Latency</th>
                      <th style={styles.mth}>P50</th>
                      <th style={styles.mth}>P95</th>
                      <th style={styles.mth}>P99</th>
                      <th style={styles.mth}>Mean</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td style={styles.mtd}><strong>Inference (ms)</strong></td>
                      <td style={{
                        ...styles.mtd,
                        color: metrics.latency.p50_ms < 8 ? "#4CAF50" : "#e53935",
                        fontWeight: 600,
                      }}>{metrics.latency.p50_ms.toFixed(3)}</td>
                      <td style={styles.mtd}>{metrics.latency.p95_ms.toFixed(3)}</td>
                      <td style={{
                        ...styles.mtd,
                        color: metrics.latency.p99_ms < 12 ? "#4CAF50" : "#e53935",
                        fontWeight: 600,
                      }}>{metrics.latency.p99_ms.toFixed(3)}</td>
                      <td style={styles.mtd}>{metrics.latency.mean_ms.toFixed(3)}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      )}

      {/* ── Videos Tab ── */}
      {tab === "videos" && (
        <div>
          <div style={{ display: "flex", gap: 12, marginBottom: 16, alignItems: "center" }}>
            <button onClick={loadVideos} style={styles.btnPrimary} disabled={loading}>
              {loading ? "Loading..." : "Refresh"}
            </button>
            <button
              onClick={deleteSelected}
              style={{ ...styles.btnDanger, opacity: selected.size === 0 ? 0.5 : 1 }}
              disabled={selected.size === 0}
            >
              Delete Selected ({selected.size})
            </button>
            {message && <span style={{ color: "#555", fontSize: 14 }}>{message}</span>}
          </div>

          {videos.length === 0 ? (
            <p style={{ color: "#888" }}>{loading ? "Loading videos..." : "No videos in the database."}</p>
          ) : (
            <div style={{ overflowX: "auto" }}>
              <table style={styles.table}>
                <thead>
                  <tr style={{ backgroundColor: "#f5f5f5" }}>
                    <th style={styles.th}>
                      <input
                        type="checkbox"
                        checked={selected.size === videos.length && videos.length > 0}
                        onChange={toggleSelectAll}
                      />
                    </th>
                    <th style={styles.th}>Title</th>
                    <th style={styles.th}>Channel</th>
                    <th style={styles.th}>Views</th>
                    <th style={styles.th}>Duration</th>
                    <th style={styles.th}>Extracted At</th>
                    <th style={styles.th}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {videos.map((video, i) => (
                    <tr
                      key={video.id}
                      style={{
                        backgroundColor: selected.has(video.id) ? "#e8f0fe" : i % 2 === 0 ? "#fff" : "#fafafa",
                      }}
                    >
                      <td style={styles.td}>
                        <input
                          type="checkbox"
                          checked={selected.has(video.id)}
                          onChange={() => toggleSelect(video.id)}
                        />
                      </td>
                      <td style={{ ...styles.td, maxWidth: 360, wordBreak: "break-word" }}>
                        {video.title || "\u2014"}
                      </td>
                      <td style={styles.td}>{video.channel_name || "\u2014"}</td>
                      <td style={styles.td}>{video.view_count || "\u2014"}</td>
                      <td style={styles.td}>{video.duration || "\u2014"}</td>
                      <td style={{ ...styles.td, whiteSpace: "nowrap", fontSize: 13, color: "#666" }}>
                        {new Date(video.extracted_at).toLocaleString()}
                      </td>
                      <td style={styles.td}>
                        <button
                          onClick={() => deleteVideo(video.id)}
                          style={styles.btnDangerSm}
                        >
                          Remove
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  loginWrap: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    height: "100vh",
    backgroundColor: "#f5f5f5",
  },
  loginBox: {
    backgroundColor: "#fff",
    padding: "40px",
    borderRadius: "8px",
    boxShadow: "0 4px 16px rgba(0,0,0,0.12)",
    minWidth: 320,
    textAlign: "center",
  },
  input: {
    display: "block",
    width: "100%",
    padding: "10px 12px",
    marginBottom: 12,
    border: "1px solid #ccc",
    borderRadius: 4,
    fontSize: 15,
    boxSizing: "border-box",
  },
  tabBar: {
    display: "flex",
    gap: 0,
    borderBottom: "2px solid #e0e0e0",
    marginBottom: 24,
  },
  tab: {
    padding: "10px 24px",
    border: "none",
    background: "none",
    cursor: "pointer",
    fontSize: 14,
    fontWeight: 500,
    color: "#888",
    borderBottom: "2px solid transparent",
    marginBottom: -2,
  },
  tabActive: {
    padding: "10px 24px",
    border: "none",
    background: "none",
    cursor: "pointer",
    fontSize: 14,
    fontWeight: 600,
    color: "#1976D2",
    borderBottom: "2px solid #1976D2",
    marginBottom: -2,
  },
  cardRow: {
    display: "flex",
    gap: 16,
    marginBottom: 24,
    flexWrap: "wrap" as const,
  },
  metricCard: {
    backgroundColor: "#fff",
    border: "1px solid #e0e0e0",
    borderRadius: 8,
    padding: "16px 20px",
    minWidth: 150,
    flex: "1 1 0",
    textAlign: "center" as const,
  },
  chartRow: {
    display: "flex",
    gap: 24,
    marginBottom: 24,
    flexWrap: "wrap" as const,
  },
  chartCard: {
    backgroundColor: "#fff",
    border: "1px solid #e0e0e0",
    borderRadius: 8,
    padding: "16px 20px",
    flex: "1 1 400px",
    overflowX: "auto" as const,
  },
  chartTitle: {
    margin: "0 0 12px 0",
    fontSize: 15,
    fontWeight: 600,
    color: "#333",
  },
  descBox: {
    backgroundColor: "#f8f9fa",
    border: "1px solid #e0e0e0",
    borderRadius: 8,
    padding: "16px 20px",
    marginBottom: 24,
    fontSize: 13,
    color: "#555",
    lineHeight: 1.6,
  },
  errorBox: {
    backgroundColor: "#fff3f3",
    border: "1px solid #e53935",
    borderRadius: 6,
    padding: "12px 16px",
    color: "#c62828",
    fontSize: 14,
    marginBottom: 16,
  },
  metricsTable: {
    width: "100%",
    borderCollapse: "collapse" as const,
    fontSize: 13,
  },
  mth: {
    padding: "8px 14px",
    textAlign: "left" as const,
    borderBottom: "2px solid #e0e0e0",
    fontWeight: 600,
    fontSize: 12,
    color: "#666",
    textTransform: "uppercase" as const,
  },
  mtd: {
    padding: "8px 14px",
    borderBottom: "1px solid #eee",
  },
  btnPrimary: {
    padding: "9px 20px",
    backgroundColor: "#1976D2",
    color: "#fff",
    border: "none",
    borderRadius: 4,
    cursor: "pointer",
    fontSize: 14,
  },
  btnSecondary: {
    padding: "9px 20px",
    backgroundColor: "#757575",
    color: "#fff",
    border: "none",
    borderRadius: 4,
    cursor: "pointer",
    fontSize: 14,
  },
  btnDanger: {
    padding: "9px 20px",
    backgroundColor: "#e53935",
    color: "#fff",
    border: "none",
    borderRadius: 4,
    cursor: "pointer",
    fontSize: 14,
  },
  btnDangerSm: {
    padding: "5px 12px",
    backgroundColor: "#e53935",
    color: "#fff",
    border: "none",
    borderRadius: 4,
    cursor: "pointer",
    fontSize: 13,
  },
  table: {
    width: "100%",
    borderCollapse: "collapse" as const,
    fontSize: 14,
    border: "1px solid #ddd",
  },
  th: {
    padding: "12px 14px",
    textAlign: "left" as const,
    borderBottom: "2px solid #ddd",
    fontWeight: 600,
    whiteSpace: "nowrap" as const,
  },
  td: {
    padding: "10px 14px",
    borderBottom: "1px solid #eee",
    verticalAlign: "middle" as const,
  },
};
