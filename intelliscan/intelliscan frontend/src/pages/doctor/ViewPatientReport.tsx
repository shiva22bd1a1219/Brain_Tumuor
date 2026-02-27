import React, { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { toast } from 'react-toastify';
import { ChevronLeft, Save } from 'lucide-react';
import { getReportById, addDoctorNotes } from '../../services/api';
import type { Report } from '../../services/api';
import TumorAnalysisResult from '../../components/TumorAnalysisResult';
import PatientInfo from '../../components/PatientInfo';

const ViewPatientReport: React.FC = () => {
  const { patientId, reportId } = useParams<{ patientId: string; reportId: string }>();
  const navigate = useNavigate();
  const [report, setReport] = useState<Report | null>(null);
  const [doctorNotes, setDoctorNotes] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  
  useEffect(() => {
    const fetchReport = async () => {
      if (!reportId) return;
      
      try {
        const fetchedReport = await getReportById(reportId);
        setReport(fetchedReport);
        if (fetchedReport?.doctorNotes) {
          setDoctorNotes(fetchedReport.doctorNotes);
        }
      } catch (error) {
        console.error('Error fetching report:', error);
        toast.error('Failed to load report data');
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchReport();
  }, [reportId]);
  
  const handleSubmitNotes = async () => {
    if (!reportId) return;
    
    setIsSubmitting(true);
    
    try {
      const updatedReport = await addDoctorNotes(reportId, doctorNotes);
      setReport(updatedReport);
      toast.success('Report updated successfully');
    } catch (error) {
      console.error('Error updating report:', error);
      toast.error('Failed to update report');
    } finally {
      setIsSubmitting(false);
    }
  };
  
  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }
  
  if (!report) {
    return (
      <div className="bg-white rounded-xl shadow-md p-6 text-center">
        <h2 className="text-xl font-semibold text-gray-800 mb-2">Report Not Found</h2>
        <p className="text-gray-600 mb-4">
          The report you are looking for does not exist or you don't have permission to view it.
        </p>
        <Link
          to="/doctor/reports"
          className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
        >
          <ChevronLeft className="h-4 w-4 mr-1" />
          Back to Reports
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
            to="/doctor/reports"
            className="inline-flex items-center text-sm text-blue-600 hover:text-blue-800"
          >
            <ChevronLeft className="h-4 w-4 mr-1" />
            Back to Reports
          </Link>
          <h1 className="text-2xl font-bold text-gray-900 mt-2">Patient Report Review</h1>
          <p className="text-gray-600">
            {report.patientName} • Report ID: {reportId} • 
            {new Date(report.createdAt).toLocaleDateString()}
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
      
      {/* Doctor's Notes */}
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
          
          <div className="space-y-4">
            <textarea
              value={doctorNotes}
              onChange={(e) => setDoctorNotes(e.target.value)}
              placeholder="Add your medical assessment, recommendations, and follow-up instructions..."
              rows={6}
              className="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border border-gray-300 rounded-md p-3"
            ></textarea>
            
            <div className="flex justify-end space-x-3">
              <button
                type="button"
                onClick={() => navigate('/doctor/reports')}
                className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleSubmitNotes}
                disabled={isSubmitting}
                className={`inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 ${
                  isSubmitting ? 'opacity-70 cursor-not-allowed' : ''
                }`}
              >
                <Save className="h-4 w-4 mr-2" />
                {isSubmitting ? 'Saving...' : 'Save Notes'}
              </button>
            </div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default ViewPatientReport;