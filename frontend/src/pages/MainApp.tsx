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

interface VideoData {
  id: number;
  title: string;
  channel_name: string;
  view_count: string;
  duration: string;
  extracted_at: string;
}

export default function MainApp() {
  const navigate = useNavigate();
  const [status, setStatus] = useState("Checking backend connection...");
  const [events, setEvents] = useState<BrowseEvent[]>([]);
  const [videos, setVideos] = useState<VideoData[]>([]);
  const [activeTab, setActiveTab] = useState<
    "live" | "clicks" | "impressions" | "videos"
  >("live");
  const [stats, setStats] = useState({
    impressions: 0,
    clicks: 0,
    sessions: 0,
  });

  // Check backend
  useEffect(() => {
    api
      .health()
      .then((res) => setStatus(res.data.message))
      .catch((err) => setStatus(`Backend error: ${err.message}`));
  }, []);

  // Poll for events
  const loadData = () => {
    api
      .getEvents(undefined, 200)
      .then((res) => {
        const evts = res.data.events || [];
        setEvents(evts);
        setStats({
          impressions: evts.filter(
            (e: BrowseEvent) => e.event_type === "impression"
          ).length,
          clicks: evts.filter((e: BrowseEvent) => e.event_type === "click")
            .length,
          sessions: new Set(evts.map((e: BrowseEvent) => e.session_id)).size,
        });
      })
      .catch(() => {});

    api
      .getVideos()
      .then((res) => setVideos(res.data.videos || []))
      .catch(() => {});
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, []);

  const filteredEvents =
    activeTab === "clicks"
      ? events.filter((e) => e.event_type === "click")
      : activeTab === "impressions"
        ? events.filter((e) => e.event_type === "impression")
        : events;

  return (
    <div style={{ padding: "20px", maxWidth: "1200px", margin: "0 auto" }}>
      <h1 style={{ textAlign: "center" }}>TwinTube Vector</h1>

      {/* Status + stats */}
      <div
        style={{
          padding: "20px",
          backgroundColor: "#f5f5f5",
          borderRadius: "8px",
          marginBottom: "20px",
        }}
      >
        <p>
          <strong>Status:</strong> {status}
        </p>

        <div style={{ display: "flex", gap: "30px", margin: "15px 0" }}>
          <div>
            <div style={{ fontSize: "28px", fontWeight: "bold" }}>
              {stats.impressions}
            </div>
            <div style={{ color: "#666" }}>Impressions</div>
          </div>
          <div>
            <div style={{ fontSize: "28px", fontWeight: "bold", color: "#4CAF50" }}>
              {stats.clicks}
            </div>
            <div style={{ color: "#666" }}>Clicks</div>
          </div>
          <div>
            <div style={{ fontSize: "28px", fontWeight: "bold", color: "#2196F3" }}>
              {stats.sessions}
            </div>
            <div style={{ color: "#666" }}>Sessions</div>
          </div>
          <div>
            <div style={{ fontSize: "28px", fontWeight: "bold", color: "#FF9800" }}>
              {videos.length}
            </div>
            <div style={{ color: "#666" }}>Videos in DB</div>
          </div>
        </div>

        {events.length === 0 && (
          <div
            style={{
              padding: "15px",
              backgroundColor: "#fff3cd",
              borderRadius: "4px",
              border: "1px solid #ffc107",
              marginTop: "10px",
            }}
          >
            <strong>No events yet.</strong> Install the Chrome extension to
            start tracking:
            <ol style={{ margin: "8px 0 0 0", paddingLeft: "20px" }}>
              <li>
                Open <code>chrome://extensions</code>
              </li>
              <li>Enable "Developer mode" (top right)</li>
              <li>Click "Load unpacked"</li>
              <li>
                Select the <code>extension/</code> folder in this project
              </li>
              <li>Browse YouTube normally</li>
            </ol>
          </div>
        )}

        {/* Action buttons */}
        <div style={{ display: "flex", gap: "10px", marginTop: "15px" }}>
          <button
            onClick={loadData}
            style={{
              padding: "10px 20px",
              backgroundColor: "#2196F3",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer",
            }}
          >
            Refresh
          </button>
          <button
            onClick={() => navigate("/admin")}
            style={{
              padding: "10px 20px",
              backgroundColor: "#757575",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer",
            }}
          >
            Admin
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: "0", marginBottom: "15px" }}>
        {(["live", "clicks", "impressions", "videos"] as const).map((tab) => (
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
                tab === "live"
                  ? "4px 0 0 4px"
                  : tab === "videos"
                    ? "0 4px 4px 0"
                    : "0",
            }}
          >
            {tab === "live"
              ? "Live Feed"
              : tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Events list */}
      {activeTab !== "videos" ? (
        <div>
          <h2>
            {activeTab === "live"
              ? "All Events"
              : activeTab === "clicks"
                ? "Clicked Videos"
                : "Impressions"}{" "}
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
                      event.event_type === "click"
                        ? "#4CAF50"
                        : event.event_type === "watch_end"
                          ? "#FF9800"
                          : "#e0e0e0"
                    }`,
                    borderRadius: "8px",
                    display: "flex",
                    alignItems: "center",
                    gap: "15px",
                  }}
                >
                  <span
                    style={{
                      padding: "4px 10px",
                      borderRadius: "12px",
                      fontSize: "12px",
                      fontWeight: "bold",
                      color: "white",
                      backgroundColor:
                        event.event_type === "click"
                          ? "#4CAF50"
                          : event.event_type === "watch_end"
                            ? "#FF9800"
                            : "#9E9E9E",
                    }}
                  >
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
      ) : (
        /* Videos tab */
        <div>
          <h2>Videos in Database ({videos.length})</h2>
          {videos.length === 0 ? (
            <p>No videos in database yet.</p>
          ) : (
            <div style={{ display: "grid", gap: "10px" }}>
              {videos.map((video) => (
                <div
                  key={video.id}
                  style={{
                    padding: "12px 15px",
                    backgroundColor: "white",
                    border: "1px solid #ddd",
                    borderRadius: "8px",
                  }}
                >
                  <div style={{ fontWeight: "bold" }}>
                    {video.title || "Unknown"}
                  </div>
                  <div style={{ color: "#666", fontSize: "13px" }}>
                    {video.channel_name || "Unknown"}{" "}
                    {video.view_count ? ` \u00B7 ${video.view_count}` : ""}{" "}
                    {video.duration ? ` \u00B7 ${video.duration}` : ""}
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
