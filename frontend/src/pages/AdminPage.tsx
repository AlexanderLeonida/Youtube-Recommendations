import React, { useEffect, useState } from "react";
import { api } from "../services/api";

interface VideoData {
  id: number;
  title: string;
  channel_name: string;
  view_count: string;
  duration: string;
  extracted_at: string;
}

const ADMIN_PASSWORD = "password";

export default function AdminPage() {
  const [authenticated, setAuthenticated] = useState(false);
  const [passwordInput, setPasswordInput] = useState("");
  const [loginError, setLoginError] = useState("");

  const [videos, setVideos] = useState<VideoData[]>([]);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  const loadVideos = () => {
    setLoading(true);
    api.getVideos()
      .then(res => {
        if (res.data.videos) setVideos(res.data.videos);
      })
      .catch(err => console.error("Error loading videos:", err))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    if (authenticated) loadVideos();
  }, [authenticated]);

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
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}>
        <h1 style={{ margin: 0 }}>TwinTube Vector — Admin</h1>
        <button onClick={() => setAuthenticated(false)} style={styles.btnSecondary}>Logout</button>
      </div>

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
                    {video.title || "—"}
                  </td>
                  <td style={styles.td}>{video.channel_name || "—"}</td>
                  <td style={styles.td}>{video.view_count || "—"}</td>
                  <td style={styles.td}>{video.duration || "—"}</td>
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
    borderCollapse: "collapse",
    fontSize: 14,
    border: "1px solid #ddd",
  },
  th: {
    padding: "12px 14px",
    textAlign: "left",
    borderBottom: "2px solid #ddd",
    fontWeight: 600,
    whiteSpace: "nowrap",
  },
  td: {
    padding: "10px 14px",
    borderBottom: "1px solid #eee",
    verticalAlign: "middle",
  },
};
