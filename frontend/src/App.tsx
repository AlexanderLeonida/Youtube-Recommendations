import React from "react";
import { BrowserRouter as Router, Routes, Route, useNavigate } from "react-router-dom";
import Tutorial from "./pages/Tutorial";
import MainApp from "./pages/MainApp";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Tutorial />} />
        <Route path="/app" element={<MainApp />} />
      </Routes>
    </Router>
  );
}
