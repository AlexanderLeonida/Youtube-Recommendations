import React, { useEffect, useState } from "react";
import axios from "axios";

export default function MainApp() {
  const [status, setStatus] = useState("Checking backend connection...");

  useEffect(() => {
    axios.get("http://localhost:4000/api/health")
      .then(res => setStatus(res.data.status))
      .catch(() => setStatus("Could not connect to backend"));
  }, []);

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>Main Application</h1>
      <p>{status}</p>
      <button onClick={() => window.location.href = "https://youtube.com"}>Go to YouTube</button>
    </div>
  );
}
