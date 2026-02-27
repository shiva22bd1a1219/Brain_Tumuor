import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useAuth } from '../../contexts/AuthContext';
import { getReportById, Report } from '../../services/api';
import { ChevronLeft } from 'lucide-react';
import PatientInfo from '../../components/PatientInfo';
import TumorAnalysisResult from '../../components/TumorAnalysisResult';
import ReportPDF from '../../components/ReportPDF';

const ViewReport: React.FC = () => {
  const { reportId } = useParams<{ reportId: string }>();
  const { user } = useAuth();
  const [report, setReport] = useState<Report | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  
  useEffect(() => {
    const fetchReport = async () => {
      if (!reportId) return;
      
      try {
        const fetchedReport = await getReportById(reportId);
        setReport(fetchedReport);
      } catch (error) {
        console.error('Error fetching report:', error);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchReport();
  }, [reportId]);
  
  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }
  
  if (!report || !user) {
    return (
      <div className="bg-white rounded-xl shadow-md p-6 text-center">
        <h2 className="text-xl font-semibold text-gray-800 mb-2">Report Not Found</h2>
        <p className="text-gray-600 mb-4">
          The report you are looking for does not exist or you don't have permission to view it.
        </p>
        <Link
          to="/patient/dashboard"
          className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
        >
          <ChevronLeft className="h-4 w-4 mr-1" />
          Back to Dashboard
        </Link>
      </div>
    );
  }
  
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className="mb-6 flex items-center justify-between">
        <div>
          <Link
            to="/patient/dashboard"
            className="inline-flex items-center text-sm text-blue-600 hover:text-blue-800"
          >
            <ChevronLeft className="h-4 w-4 mr-1" />
            Back to Dashboard
          </Link>
          <h1 className="text-2xl font-bold text-gray-900 mt-2">MRI Scan Report</h1>
          <p className="text-gray-600">
            Report ID: {reportId} • Generated on {new Date(report.createdAt).toLocaleDateString()}
          </p>
        </div>
      </div>
      
      {/* Patient Information */}
      <PatientInfo patientDetails={report.patientDetails} patientName={report.patientName} />
      
      {/* Tumor Analysis Result */}
      <TumorAnalysisResult 
        classification={report.classification} 
        maskImageUrl={report.maskImageUrl} 
      />
      
      {/* Doctor's Notes (if available) */}
      {report.doctorNotes && (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="bg-white rounded-xl shadow-md overflow-hidden mt-6"
        >
          <div className="p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              Doctor's Notes
            </h2>
            <div className="bg-gray-50 p-4 rounded-lg">
              <p className="text-gray-700">{report.doctorNotes}</p>
            </div>
          </div>
        </motion.div>
      )}
      
      {/* PDF Report */}
      <ReportPDF report={report} />
    </motion.div>
  );
};

export default ViewReport;