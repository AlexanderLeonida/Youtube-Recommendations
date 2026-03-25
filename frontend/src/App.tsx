import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Tutorial from "./pages/Tutorial";
import MainApp from "./pages/MainApp";
import AdminPage from "./pages/AdminPage";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Tutorial />} />
        <Route path="/app" element={<MainApp />} />
        <Route path="/admin" element={<AdminPage />} />
      </Routes>
    </Router>
  );
}
