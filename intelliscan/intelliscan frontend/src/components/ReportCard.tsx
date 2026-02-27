import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FileText, Clock, CheckCircle } from 'lucide-react';
import { Report } from '../services/api';

interface ReportCardProps {
  report: Report;
  isDoctor?: boolean;
}

const ReportCard: React.FC<ReportCardProps> = ({ report, isDoctor = false }) => {
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };
  
  const getStatusBadge = () => {
    if (report.status === 'reviewed') {
      return (
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
          <CheckCircle className="w-3 h-3 mr-1" />
          Reviewed
        </span>
      );
    }
    
    return (
      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
        <Clock className="w-3 h-3 mr-1" />
        Pending Review
      </span>
    );
  };
  
  const reportUrl = isDoctor 
    ? `/doctor/reports/${report.patientId}/${report.id}` 
    : `/patient/reports/${report.id}`;
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow"
    >
      <div className="p-5">
        <div className="flex justify-between items-start">
          <div className="flex items-center">
            <FileText className="w-5 h-5 text-blue-500 mr-2" />
            <h3 className="text-lg font-medium text-gray-800">
              {report.classification.label} Report
            </h3>
          </div>
          {getStatusBadge()}
        </div>
        
        <div className="mt-3 text-sm text-gray-500">
          <p className="flex items-center">
            <Clock className="w-4 h-4 mr-1" />
            {formatDate(report.createdAt)}
          </p>
        </div>
        
        <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
          <div className="bg-gray-50 p-2 rounded">
            <p className="text-gray-500 text-xs">Type</p>
            <p className="font-medium text-gray-900 capitalize">{report.classification.label}</p>
          </div>
          <div className="bg-gray-50 p-2 rounded">
            <p className="text-gray-500 text-xs">Confidence</p>
            <p className="font-medium text-gray-900">{(report.classification.confidence * 100).toFixed(1)}%</p>
          </div>
        </div>
        
        <div className="mt-4 pt-3 border-t border-gray-100 flex justify-end">
          <Link
            to={reportUrl}
            className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            View Report
          </Link>
        </div>
      </div>
    </motion.div>
  );
};

export default ReportCard;