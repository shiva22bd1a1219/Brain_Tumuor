import React from 'react';
import { motion } from 'framer-motion';
import { User, Calendar, Phone, Mail, MapPin } from 'lucide-react';
import { Report } from '../services/api';

interface PatientInfoProps {
  patientDetails: Report['patientDetails'];
  patientName: string;
}

const PatientInfo: React.FC<PatientInfoProps> = ({ patientDetails, patientName }) => {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="bg-white rounded-xl shadow-md overflow-hidden"
    >
      <div className="p-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
          <User className="w-5 h-5 mr-2 text-blue-500" />
          Patient Information
        </h2>
        
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex items-start">
              <div className="flex-shrink-0 bg-blue-100 p-2 rounded-md">
                <User className="w-5 h-5 text-blue-600" />
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-500">Full Name</p>
                <p className="text-base font-medium text-gray-800">{patientName}</p>
              </div>
            </div>
            
            <div className="flex items-start">
              <div className="flex-shrink-0 bg-blue-100 p-2 rounded-md">
                <Mail className="w-5 h-5 text-blue-600" />
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-500">Email Address</p>
                <p className="text-base font-medium text-gray-800">{patientDetails.email}</p>
              </div>
            </div>
            
            <div className="flex items-start">
              <div className="flex-shrink-0 bg-blue-100 p-2 rounded-md">
                <Calendar className="w-5 h-5 text-blue-600" />
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-500">Age</p>
                <p className="text-base font-medium text-gray-800">{patientDetails.age || 'Not provided'}</p>
              </div>
            </div>
            
            <div className="flex items-start">
              <div className="flex-shrink-0 bg-blue-100 p-2 rounded-md">
                <User className="w-5 h-5 text-blue-600" />
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-500">Gender</p>
                <p className="text-base font-medium text-gray-800">{patientDetails.gender || 'Not provided'}</p>
              </div>
            </div>
            
            <div className="flex items-start">
              <div className="flex-shrink-0 bg-blue-100 p-2 rounded-md">
                <Phone className="w-5 h-5 text-blue-600" />
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-500">Contact Number</p>
                <p className="text-base font-medium text-gray-800">{patientDetails.contactNumber || 'Not provided'}</p>
              </div>
            </div>
            
            <div className="flex items-start">
              <div className="flex-shrink-0 bg-blue-100 p-2 rounded-md">
                <Calendar className="w-5 h-5 text-blue-600" />
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-500">Registration Date</p>
                <p className="text-base font-medium text-gray-800">{patientDetails.registrationDate}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default PatientInfo;