import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { useAuth } from "./contexts/AuthContext";
import { motion } from "framer-motion";

// Layouts
import MainLayout from "./layouts/MainLayout";

// Pages
import Login from "./pages/Login";
import Register from "./pages/Register";
import PatientDashboard from "./pages/patient/Dashboard";
import PatientProfile from "./pages/patient/Profile";
import UploadScan from "./pages/patient/UploadScan";
import ViewReport from "./pages/patient/ViewReport";
import DoctorDashboard from "./pages/doctor/Dashboard";
import DoctorReports from "./pages/doctor/Reports";
import ViewPatientReport from "./pages/doctor/ViewPatientReport";
import NotFound from "./pages/NotFound";

// Protected Route Component
const ProtectedRoute = ({
  children,
  userType,
}: {
  children: React.ReactNode;
  userType?: string;
}) => {
  const { user } = useAuth();

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  if (userType && user.role !== userType) {
    return (
      <Navigate
        to={
          user.role === "patient" ? "/patient/dashboard" : "/doctor/dashboard"
        }
        replace
      />
    );
  }

  return <>{children}</>;
};

function App() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="min-h-screen bg-gray-50"
    >
      <Routes>
        {/* Public Routes */}
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />

        {/* Patient Routes */}
        <Route
          path="/patient"
          element={
            <ProtectedRoute userType="patient">
              <MainLayout />
            </ProtectedRoute>
          }
        >
          <Route path="dashboard" element={<PatientDashboard />} />
          <Route path="profile" element={<PatientProfile />} />
          <Route path="upload" element={<UploadScan />} />
          <Route path="reports/:reportId" element={<ViewReport />} />
        </Route>

        {/* Doctor Routes */}
        <Route
          path="/doctor"
          element={
            <ProtectedRoute userType="doctor">
              <MainLayout />
            </ProtectedRoute>
          }
        >
          <Route path="dashboard" element={<DoctorDashboard />} />
          <Route path="reports" element={<DoctorReports />} />
          <Route
            path="reports/:patientId/:reportId"
            element={<ViewPatientReport />}
          />
        </Route>

        {/* Redirect root to login */}
        <Route path="/" element={<Navigate to="/login" />} />

        {/* 404 Not Found */}
        <Route path="*" element={<NotFound />} />
      </Routes>
    </motion.div>
  );
}

export default App;
