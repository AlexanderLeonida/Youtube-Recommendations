import React from "react";
import { useNavigate } from "react-router-dom";

export default function Tutorial() {
  const navigate = useNavigate();
  
  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>Welcome to the YouTube Recommendations AI App!</h1>
      <p>Walk through this short tutorial before getting started.</p>
      <button onClick={() => navigate("/app")}>Finish Tutorial</button>
    </div>
  );
}
