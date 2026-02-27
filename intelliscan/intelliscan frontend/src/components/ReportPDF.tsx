import React, { useRef } from 'react';
import { toast } from 'react-toastify';
import { FileDown, Printer } from 'lucide-react';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import { Report } from '../services/api';

interface ReportPDFProps {
  report: Report;
}

const ReportPDF: React.FC<ReportPDFProps> = ({ report }) => {
  const reportRef = useRef<HTMLDivElement>(null);
  
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  };
  
  const generatePDF = async () => {
    if (!reportRef.current) return;
    
    try {
      toast.info('Generating PDF report...');
      
      const canvas = await html2canvas(reportRef.current, {
        scale: 2,
        useCORS: true,
        allowTaint: true,
      });
      
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      const imgProps = pdf.getImageProperties(imgData);
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;
      
      pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
      pdf.save(`brain_tumor_report_${report.id}.pdf`);
      
      toast.success('PDF report generated successfully!');
    } catch (error) {
      console.error('Error generating PDF:', error);
      toast.error('Failed to generate PDF. Please try again.');
    }
  };
  
  const handlePrint = () => {
    window.print();
  };
  
  return (
    <div className="mt-6">
      <div className="flex justify-end space-x-3 mb-4 print:hidden">
        <button
          onClick={handlePrint}
          className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
        >
          <Printer className="h-4 w-4 mr-2" />
          Print Report
        </button>
        <button
          onClick={generatePDF}
          className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
        >
          <FileDown className="h-4 w-4 mr-2" />
          Download PDF
        </button>
      </div>
      
      <div 
        ref={reportRef} 
        className="bg-white rounded-xl shadow-md overflow-hidden p-8 print:shadow-none print:p-0"
        style={{ maxWidth: '800px', margin: '0 auto' }}
      >
        {/* Report Header */}
        <div className="border-b border-gray-200 pb-6 mb-6">
          <div className="flex justify-between items-center">
            <div className="flex items-center">
              <div className="text-blue-600 mr-2">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M2 12c0 5.523 4.477 10 10 10s10-4.477 10-10S17.523 2 12 2 2 6.477 2 12z"/>
                  <path d="M12 2v10l4.56 4.56"/>
                  <path d="M12 12l5.5-2"/>
                </svg>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">IntelliScan</h1>
                <p className="text-sm text-gray-500">Brain Tumor Analysis Report</p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm text-gray-500">Report ID: {report.id}</p>
              <p className="text-sm text-gray-500">Date: {formatDate(report.createdAt)}</p>
            </div>
          </div>
        </div>
        
        {/* Patient Information */}
        <div className="mb-8">
          <h2 className="text-lg font-semibold mb-4 text-gray-800 border-b pb-2">Patient Information</h2>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-500">Patient Name</p>
              <p className="font-medium">{report.patientName}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Patient ID</p>
              <p className="font-medium">{report.patientId}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Age</p>
              <p className="font-medium">{report.patientDetails.age || 'Not provided'}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Gender</p>
              <p className="font-medium">{report.patientDetails.gender || 'Not provided'}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Contact</p>
              <p className="font-medium">{report.patientDetails.contactNumber || 'Not provided'}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Email</p>
              <p className="font-medium">{report.patientDetails.email}</p>
            </div>
          </div>
        </div>
        
        {/* Analysis Results */}
        <div className="mb-8">
          <h2 className="text-lg font-semibold mb-4 text-gray-800 border-b pb-2">Analysis Results</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Left Column - Result Summary */}
            <div>
              <div className="mb-6">
                <h3 className="text-md font-medium mb-2">Classification</h3>
                <div className="bg-blue-50 p-4 rounded-lg">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-gray-700">Detected Type:</span>
                    <span className="font-bold text-gray-900 capitalize">{report.classification.label}</span>
                  </div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-gray-700">Confidence:</span>
                    <span className="font-bold text-gray-900">{(report.classification.confidence * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
              
              <div>
                <h3 className="text-md font-medium mb-2">Probability Distribution</h3>
                <div className="space-y-3">
                  {Object.entries(report.classification.probabilities).map(([label, prob]) => (
                    <div key={label} className="relative pt-1">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-xs font-medium text-gray-600 uppercase">{label}</span>
                        <span className="text-xs font-medium text-gray-800">{(prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="overflow-hidden h-2 text-xs flex rounded bg-gray-200">
                        <div 
                          style={{ width: `${prob * 100}%` }}
                          className={`shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center ${
                            label === report.classification.label ? 'bg-blue-500' : 'bg-gray-400'
                          }`}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            
            {/* Right Column - Segmentation */}
            <div>
              <h3 className="text-md font-medium mb-2">Tumor Segmentation</h3>
              <div className="border border-gray-200 rounded-lg overflow-hidden">
                <img 
                  src={report.maskImageUrl} 
                  alt="Tumor segmentation" 
                  className="w-full h-auto"
                  crossOrigin="anonymous"
                />
              </div>
              <p className="mt-2 text-sm text-gray-600">
                The green overlay represents the detected tumor region in the MRI scan.
              </p>
            </div>
          </div>
        </div>
        
        {/* Doctor Notes (if available) */}
        {report.doctorNotes && (
          <div className="mb-8">
            <h2 className="text-lg font-semibold mb-4 text-gray-800 border-b pb-2">Doctor's Notes</h2>
            <div className="bg-gray-50 p-4 rounded-lg">
              <p className="text-gray-700">{report.doctorNotes}</p>
            </div>
          </div>
        )}
        
        {/* Footer */}
        <div className="mt-10 pt-6 border-t border-gray-200 text-center">
          <p className="text-sm text-gray-500">
            This report was generated by IntelliScan AI. Results should be reviewed by a qualified medical professional.
          </p>
          <p className="text-sm text-gray-400 mt-1">
            Report generated on {new Date().toLocaleDateString()}
          </p>
        </div>
      </div>
    </div>
  );
};

export default ReportPDF;