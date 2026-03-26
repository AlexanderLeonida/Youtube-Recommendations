import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../services/api";

interface BrowseEvent {
  id: number;
  event_type: "impression" | "click" | "watch_end";
  session_id: string;
  video_id: string;
  title: string;
  channel_name: string;
  views: string;
  duration: string;
  watch_duration_sec: number | null;
  created_at: string;
}

interface Recommendation {
  video_id: string;
  score: number;
  ctr: number | null;
  youtube_url: string | null;
  title?: string;
  channel?: string;
  views?: string;
  duration?: string;
}

interface CTRStats {
  overall_ctr: number;
  total_impressions: number;
  total_clicks: number;
  unique_videos: number;
}

export default function MainApp() {
  const navigate = useNavigate();
  const [status, setStatus] = useState("Checking connection...");
  const [events, setEvents] = useState<BrowseEvent[]>([]);
  const [stats, setStats] = useState({ impressions: 0, clicks: 0, sessions: 0 });
  const [activeTab, setActiveTab] = useState<"live" | "clicks" | "recommendations">("live");

  // ML state
  const [isTraining, setIsTraining] = useState(false);
  const [trainResult, setTrainResult] = useState<string | null>(null);
  const [modelReady, setModelReady] = useState(false);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [isLoadingRecs, setIsLoadingRecs] = useState(false);
  const [ctrStats, setCtrStats] = useState<CTRStats | null>(null);

  // Check backend + ML status
  useEffect(() => {
    api.health()
      .then((res) => setStatus(res.data.message))
      .catch((err) => setStatus(`Backend error: ${err.message}`));

    api.getTrainStatus()
      .then((res) => {
        setModelReady(res.data.model_exists && res.data.index_exists);
        if (res.data.ctr_stats) setCtrStats(res.data.ctr_stats);
      })
      .catch(() => {});
  }, []);

  // Poll events
  const loadData = () => {
    api.getEvents(undefined, 200).then((res) => {
      const evts = res.data.events || [];
      setEvents(evts);
      setStats({
        impressions: evts.filter((e: BrowseEvent) => e.event_type === "impression").length,
        clicks: evts.filter((e: BrowseEvent) => e.event_type === "click").length,
        sessions: new Set(evts.map((e: BrowseEvent) => e.session_id)).size,
      });
    }).catch(() => {});
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, []);

  // ── Actions ──────────────────────────────────────────────────────────────

  const handleBrowseYouTube = () => {
    window.open("https://www.youtube.com", "_blank");
  };

  const handleTrain = async () => {
    setIsTraining(true);
    setTrainResult(null);
    try {
      const res = await api.trainModel(10, 32);
      setTrainResult(res.data.status === "training_started"
        ? "Training started in background. This may take a minute..."
        : res.data.message || "Training initiated.");
      // Poll for completion
      const poll = setInterval(async () => {
        try {
          const s = await api.getTrainStatus();
          if (s.data.model_exists && s.data.index_exists) {
            setModelReady(true);
            setIsTraining(false);
            if (s.data.ctr_stats) setCtrStats(s.data.ctr_stats);
            const ctrMsg = s.data.ctr_stats
              ? ` Overall CTR: ${(s.data.ctr_stats.overall_ctr * 100).toFixed(1)}%`
              : "";
            setTrainResult(`Training complete! Model is ready.${ctrMsg}`);
            clearInterval(poll);
          }
        } catch {}
      }, 5000);
      // Stop polling after 5 minutes max
      setTimeout(() => { clearInterval(poll); setIsTraining(false); }, 300000);
    } catch (err: any) {
      setTrainResult(`Training failed: ${err.response?.data?.detail || err.response?.data?.error || err.message}`);
      setIsTraining(false);
    }
  };

  const handleRecommend = async () => {
    setIsLoadingRecs(true);
    setActiveTab("recommendations");
    try {
      const res = await api.getRecommendations(20);
      setRecommendations(res.data.recommendations || []);
      if (res.data.overall_ctr != null) {
        setCtrStats((prev) => prev ? { ...prev, overall_ctr: res.data.overall_ctr } : null);
      }
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.response?.data?.error || err.message;
      setRecommendations([]);
      setTrainResult(`Recommendation error: ${msg}`);
    } finally {
      setIsLoadingRecs(false);
    }
  };

  const filteredEvents =
    activeTab === "clicks"
      ? events.filter((e) => e.event_type === "click")
      : events;

  const tabs = ["live", "clicks", "recommendations"] as const;

  return (
    <div style={{ padding: "20px", maxWidth: "1200px", margin: "0 auto" }}>
      <h1 style={{ textAlign: "center" }}>TwinTube Vector</h1>

      {/* Status + stats panel */}
      <div style={{ padding: "20px", backgroundColor: "#f5f5f5", borderRadius: "8px", marginBottom: "20px" }}>
        <p><strong>Status:</strong> {status}</p>

        <div style={{ display: "flex", gap: "30px", margin: "15px 0" }}>
          <StatBox value={stats.impressions} label="Impressions" />
          <StatBox value={stats.clicks} label="Clicks" color="#4CAF50" />
          <StatBox value={stats.sessions} label="Sessions" color="#2196F3" />
          <StatBox
            value={modelReady ? "Ready" : "Not trained"}
            label="Model"
            color={modelReady ? "#4CAF50" : "#FF9800"}
            small
          />
          {ctrStats && (
            <StatBox
              value={`${(ctrStats.overall_ctr * 100).toFixed(1)}%`}
              label="CTR"
              color="#9C27B0"
              small
            />
          )}
        </div>

        {/* 4 main action buttons */}
        <div style={{ display: "flex", gap: "10px", flexWrap: "wrap", marginTop: "15px" }}>
          {/* 1. Browse YouTube */}
          <ActionButton
            onClick={handleBrowseYouTube}
            color="#FF0000"
            label="Browse YouTube"
          />

          {/* 2. Back to App (refresh) */}
          <ActionButton
            onClick={loadData}
            color="#2196F3"
            label="Refresh Data"
          />

          {/* 3. Retrain model */}
          <ActionButton
            onClick={handleTrain}
            color="#FF9800"
            label={isTraining ? "Training..." : "Train Model"}
            disabled={isTraining || stats.clicks < 3}
            title={stats.clicks < 3 ? "Need at least 3 clicks to train" : ""}
          />

          {/* 4. Get recommendations */}
          <ActionButton
            onClick={handleRecommend}
            color="#9C27B0"
            label={isLoadingRecs ? "Loading..." : "Get Recommendations"}
            disabled={isLoadingRecs || !modelReady}
            title={!modelReady ? "Train the model first" : ""}
          />

          <ActionButton
            onClick={() => navigate("/admin")}
            color="#757575"
            label="Admin"
          />
        </div>

        {/* Training result message */}
        {trainResult && (
          <div style={{
            marginTop: "10px",
            padding: "10px 15px",
            backgroundColor: trainResult.includes("error") || trainResult.includes("failed")
              ? "#ffebee" : "#e8f5e9",
            borderRadius: "4px",
            fontSize: "14px",
          }}>
            {trainResult}
          </div>
        )}

        {/* Extension install hint */}
        {events.length === 0 && (
          <div style={{
            padding: "15px", backgroundColor: "#fff3cd", borderRadius: "4px",
            border: "1px solid #ffc107", marginTop: "10px",
          }}>
            <strong>No events yet.</strong> Install the Chrome extension:
            <ol style={{ margin: "8px 0 0 0", paddingLeft: "20px" }}>
              <li>Open <code>chrome://extensions</code></li>
              <li>Enable "Developer mode"</li>
              <li>Click "Load unpacked" and select the <code>extension/</code> folder</li>
              <li>Click "Browse YouTube" above and start watching!</li>
            </ol>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: "0", marginBottom: "15px" }}>
        {tabs.map((tab, i) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              padding: "8px 20px",
              backgroundColor: activeTab === tab ? "#333" : "#e0e0e0",
              color: activeTab === tab ? "white" : "#333",
              border: "none",
              cursor: "pointer",
              borderRadius:
                i === 0 ? "4px 0 0 4px" :
                i === tabs.length - 1 ? "0 4px 4px 0" : "0",
            }}
          >
            {tab === "live" ? "Live Feed" :
             tab === "clicks" ? "Clicked Videos" :
             "Recommendations"}
          </button>
        ))}
      </div>

      {/* Content */}
      {activeTab === "recommendations" ? (
        <div>
          <h2>Recommendations ({recommendations.length})</h2>
          {isLoadingRecs ? (
            <p>Loading recommendations...</p>
          ) : recommendations.length === 0 ? (
            <p>No recommendations yet. Train the model and try again.</p>
          ) : (
            <div style={{ display: "grid", gap: "10px" }}>
              {recommendations.map((rec, i) => (
                <a
                  key={rec.video_id}
                  href={rec.youtube_url || "#"}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ textDecoration: "none", color: "inherit" }}
                >
                  <div style={{
                    padding: "12px 15px",
                    backgroundColor: "white",
                    border: "2px solid #9C27B0",
                    borderRadius: "8px",
                    display: "flex",
                    alignItems: "center",
                    gap: "15px",
                    cursor: "pointer",
                  }}>
                    <span style={{
                      padding: "4px 12px", borderRadius: "12px", fontSize: "14px",
                      fontWeight: "bold", color: "white", backgroundColor: "#9C27B0",
                      minWidth: "30px", textAlign: "center",
                    }}>
                      #{i + 1}
                    </span>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: "bold" }}>
                        {rec.title || rec.video_id}
                      </div>
                      <div style={{ color: "#666", fontSize: "13px" }}>
                        {rec.channel || ""}
                        {rec.views ? ` \u00B7 ${rec.views}` : ""}
                        {rec.duration ? ` \u00B7 ${rec.duration}` : ""}
                        {!rec.title && rec.youtube_url ? " \u00B7 Click to watch" : ""}
                      </div>
                    </div>
                    <div style={{ color: "#999", fontSize: "12px", textAlign: "right" }}>
                      <div>score: {rec.score.toFixed(4)}</div>
                      {rec.ctr != null && (
                        <div style={{ color: "#9C27B0" }}>
                          CTR: {(rec.ctr * 100).toFixed(1)}%
                        </div>
                      )}
                    </div>
                  </div>
                </a>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div>
          <h2>
            {activeTab === "live" ? "All Events" : "Clicked Videos"}{" "}
            ({filteredEvents.length})
          </h2>
          {filteredEvents.length === 0 ? (
            <p>No events yet. Browse YouTube with the extension installed.</p>
          ) : (
            <div style={{ display: "grid", gap: "10px" }}>
              {filteredEvents.map((event) => (
                <div
                  key={event.id}
                  style={{
                    padding: "12px 15px",
                    backgroundColor: "white",
                    border: `2px solid ${
                      event.event_type === "click" ? "#4CAF50" :
                      event.event_type === "watch_end" ? "#FF9800" : "#e0e0e0"
                    }`,
                    borderRadius: "8px",
                    display: "flex",
                    alignItems: "center",
                    gap: "15px",
                  }}
                >
                  <span style={{
                    padding: "4px 10px", borderRadius: "12px", fontSize: "12px",
                    fontWeight: "bold", color: "white",
                    backgroundColor:
                      event.event_type === "click" ? "#4CAF50" :
                      event.event_type === "watch_end" ? "#FF9800" : "#9E9E9E",
                  }}>
                    {event.event_type === "watch_end"
                      ? `watched ${event.watch_duration_sec}s`
                      : event.event_type}
                  </span>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontWeight: "bold" }}>
                      {event.title || event.video_id}
                    </div>
                    <div style={{ color: "#666", fontSize: "13px" }}>
                      {event.channel_name || ""}
                      {event.views ? ` \u00B7 ${event.views}` : ""}
                      {event.duration ? ` \u00B7 ${event.duration}` : ""}
                    </div>
                  </div>
                  <div style={{ color: "#999", fontSize: "12px" }}>
                    {new Date(event.created_at).toLocaleTimeString()}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Helper components ──────────────────────────────────────────────────────

function StatBox({ value, label, color, small }: {
  value: number | string; label: string; color?: string; small?: boolean;
}) {
  return (
    <div>
      <div style={{
        fontSize: small ? "18px" : "28px",
        fontWeight: "bold",
        color: color || "#333",
      }}>
        {value}
      </div>
      <div style={{ color: "#666" }}>{label}</div>
    </div>
  );
}

function ActionButton({ onClick, color, label, disabled, title }: {
  onClick: () => void; color: string; label: string;
  disabled?: boolean; title?: string;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      style={{
        padding: "10px 20px",
        backgroundColor: disabled ? "#ccc" : color,
        color: "white",
        border: "none",
        borderRadius: "4px",
        cursor: disabled ? "not-allowed" : "pointer",
        fontSize: "14px",
        fontWeight: "bold",
      }}
    >
      {label}
    </button>
  );
}
