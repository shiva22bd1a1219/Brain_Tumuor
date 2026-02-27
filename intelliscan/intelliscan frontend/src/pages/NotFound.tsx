import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Brain, Home } from 'lucide-react';

const NotFound: React.FC = () => {
  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="min-h-screen flex items-center justify-center bg-gray-50 p-6"
    >
      <div className="max-w-md w-full text-center">
        <Brain className="mx-auto h-16 w-16 text-blue-500" />
        <h1 className="mt-6 text-3xl font-extrabold text-gray-900">404 - Page Not Found</h1>
        <p className="mt-3 text-base text-gray-500">
          The page you are looking for doesn't exist or has been moved.
        </p>
        <div className="mt-8">
          <Link
            to="/"
            className="inline-flex items-center px-4 py-2 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            <Home className="h-5 w-5 mr-2" />
            Back to Home
          </Link>
        </div>
      </div>
    </motion.div>
  );
};

export default NotFound;