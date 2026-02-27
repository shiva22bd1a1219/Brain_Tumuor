import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useAuth } from '../../contexts/AuthContext';
import { getPatientReports, Report } from '../../services/api';
import { FileText, Upload, PlusCircle, AlertTriangle } from 'lucide-react';
import ReportCard from '../../components/ReportCard';

const Dashboard: React.FC = () => {
  const { user } = useAuth();
  const [reports, setReports] = useState<Report[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  
  useEffect(() => {
    const fetchReports = async () => {
      if (!user) return;
      
      try {
        const fetchedReports = await getPatientReports(user.id);
        setReports(fetchedReports);
      } catch (error) {
        console.error('Error fetching reports:', error);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchReports();
  }, [user]);
  
  // Group reports by status
  const pendingReports = reports.filter(report => report.status === 'pending');
  const reviewedReports = reports.filter(report => report.status === 'reviewed');
  
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Patient Dashboard</h1>
        <p className="text-gray-600 mt-1">Welcome back, {user?.name}</p>
      </div>
      
      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
        <Link
          to="/patient/upload"
          className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow"
        >
          <div className="flex items-start">
            <div className="flex-shrink-0 bg-blue-100 p-3 rounded-md">
              <Upload className="w-6 h-6 text-blue-600" />
            </div>
            <div className="ml-4">
              <h3 className="text-lg font-medium text-gray-800">Upload New Scan</h3>
              <p className="mt-1 text-sm text-gray-600">
                Upload a new MRI scan for analysis
              </p>
              <button className="mt-3 text-sm font-medium text-blue-600 flex items-center">
                Upload now
                <PlusCircle className="w-4 h-4 ml-1" />
              </button>
            </div>
          </div>
        </Link>
        
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-start">
            <div className="flex-shrink-0 bg-blue-100 p-3 rounded-md">
              <FileText className="w-6 h-6 text-blue-600" />
            </div>
            <div className="ml-4">
              <h3 className="text-lg font-medium text-gray-800">Reports Summary</h3>
              <div className="mt-1 space-y-1">
                <p className="text-sm text-gray-600">
                  Total Reports: <span className="font-medium">{reports.length}</span>
                </p>
                <p className="text-sm text-gray-600">
                  Pending Review: <span className="font-medium">{pendingReports.length}</span>
                </p>
                <p className="text-sm text-gray-600">
                  Reviewed: <span className="font-medium">{reviewedReports.length}</span>
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Reports List */}
      <div className="bg-white rounded-xl shadow-md overflow-hidden mb-6">
        <div className="p-6 border-b border-gray-200">
          <h2 className="text-lg font-medium text-gray-800">Recent Reports</h2>
        </div>
        
        <div className="p-6">
          {isLoading ? (
            <div className="flex justify-center items-center py-8">
              <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500"></div>
            </div>
          ) : reports.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {reports.map((report) => (
                <ReportCard key={report.id} report={report} />
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-gray-100">
                <AlertTriangle className="h-6 w-6 text-gray-600" />
              </div>
              <h3 className="mt-2 text-sm font-medium text-gray-900">No reports found</h3>
              <p className="mt-1 text-sm text-gray-500">
                Get started by uploading your first MRI scan.
              </p>
              <div className="mt-6">
                <Link
                  to="/patient/upload"
                  className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                >
                  <Upload className="h-4 w-4 mr-2" />
                  Upload Scan
                </Link>
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* Health Tips Section */}
      <div className="bg-gradient-to-r from-blue-50 to-blue-100 rounded-xl shadow-sm p-6 mt-8">
        <h2 className="text-lg font-medium text-gray-800 mb-4">Health Tips & Information</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <h3 className="font-medium text-gray-800 mb-2">Understanding MRI Results</h3>
            <p className="text-sm text-gray-600">
              Learn how to interpret your MRI scan results and what different classifications mean.
            </p>
            <a href="#" className="text-sm text-blue-600 hover:text-blue-800 mt-2 inline-block">
              Read more →
            </a>
          </div>
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <h3 className="font-medium text-gray-800 mb-2">Preparing for an MRI</h3>
            <p className="text-sm text-gray-600">
              Tips on how to prepare for your MRI scan appointment and what to expect during the procedure.
            </p>
            <a href="#" className="text-sm text-blue-600 hover:text-blue-800 mt-2 inline-block">
              Read more →
            </a>
          </div>
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <h3 className="font-medium text-gray-800 mb-2">Brain Health Resources</h3>
            <p className="text-sm text-gray-600">
              Explore resources and support groups for patients with neurological conditions.
            </p>
            <a href="#" className="text-sm text-blue-600 hover:text-blue-800 mt-2 inline-block">
              Read more →
            </a>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default Dashboard;