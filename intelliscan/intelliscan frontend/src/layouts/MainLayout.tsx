import React, { useState } from 'react';
import { Outlet, Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { motion } from 'framer-motion';
import { 
  Brain, 
  User, 
  FileText, 
  Upload, 
  LogOut, 
  Menu, 
  X,
  Home,
  FilePlus,
  Users,
  Bell
} from 'lucide-react';

const MainLayout: React.FC = () => {
  const { user, logout } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [notifications, setNotifications] = useState<number>(2); // Example notification count

  const isPatient = user?.role === 'patient';
  
  const patientLinks = [
    { name: 'Dashboard', path: '/patient/dashboard', icon: <Home size={20} /> },
    { name: 'Upload Scan', path: '/patient/upload', icon: <Upload size={20} /> },
    { name: 'Profile', path: '/patient/profile', icon: <User size={20} /> },
  ];
  
  const doctorLinks = [
    { name: 'Dashboard', path: '/doctor/dashboard', icon: <Home size={20} /> },
    { name: 'Patient Reports', path: '/doctor/reports', icon: <FileText size={20} /> },
  ];
  
  const navigationLinks = isPatient ? patientLinks : doctorLinks;

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar - Desktop */}
      <motion.aside 
        initial={{ x: -100, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.3 }}
        className="hidden md:flex flex-col w-64 bg-white border-r border-gray-200"
      >
        <div className="p-5 border-b border-gray-200">
          <Link to={isPatient ? '/patient/dashboard' : '/doctor/dashboard'} className="flex items-center space-x-3">
            <Brain className="h-8 w-8 text-blue-600" />
            <span className="text-xl font-bold text-gray-800">IntelliScan</span>
          </Link>
          <div className="mt-2 text-sm text-gray-500">{isPatient ? 'Patient Portal' : 'Doctor Portal'}</div>
        </div>
        
        <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
          {navigationLinks.map((link) => (
            <Link
              key={link.path}
              to={link.path}
              className={`flex items-center px-4 py-3 rounded-lg text-sm transition-colors ${
                location.pathname === link.path
                  ? 'bg-blue-50 text-blue-700'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              <span className="mr-3">{link.icon}</span>
              {link.name}
            </Link>
          ))}
        </nav>
        
        <div className="p-4 border-t border-gray-200">
          <button
            onClick={handleLogout}
            className="flex items-center w-full px-4 py-2 text-sm text-gray-600 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <LogOut size={20} className="mr-3" />
            Sign Out
          </button>
        </div>
      </motion.aside>
      
      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 shadow-sm">
          <div className="px-4 sm:px-6 py-4 flex justify-between items-center">
            {/* Mobile menu button */}
            <button 
              onClick={toggleMobileMenu}
              className="md:hidden text-gray-600 focus:outline-none"
            >
              <Menu size={24} />
            </button>
            
            <div className="md:hidden flex items-center">
              <Brain className="h-6 w-6 text-blue-600" />
              <span className="ml-2 text-lg font-semibold text-gray-800">IntelliScan</span>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Notifications */}
              <div className="relative">
                <Bell className="w-6 h-6 text-gray-600" />
                {notifications > 0 && (
                  <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs w-5 h-5 flex items-center justify-center rounded-full">
                    {notifications}
                  </span>
                )}
              </div>
              
              {/* User avatar */}
              <div className="flex items-center space-x-2">
                <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white font-semibold">
                  {user?.name?.charAt(0) || 'U'}
                </div>
                <span className="hidden sm:inline-block text-sm font-medium text-gray-700">
                  {user?.name || 'User'}
                </span>
              </div>
            </div>
          </div>
        </header>
        
        {/* Mobile Navigation Menu */}
        {isMobileMenuOpen && (
          <div className="md:hidden fixed inset-0 z-50 bg-gray-900 bg-opacity-50">
            <motion.div 
              initial={{ x: -300 }}
              animate={{ x: 0 }}
              exit={{ x: -300 }}
              transition={{ duration: 0.3 }}
              className="h-full w-64 bg-white shadow-xl"
            >
              <div className="p-4 flex justify-between items-center border-b border-gray-200">
                <div className="flex items-center">
                  <Brain className="h-6 w-6 text-blue-600" />
                  <span className="ml-2 text-lg font-semibold text-gray-800">IntelliScan</span>
                </div>
                <button onClick={toggleMobileMenu} className="text-gray-600">
                  <X size={24} />
                </button>
              </div>
              
              <nav className="p-4">
                {navigationLinks.map((link) => (
                  <Link
                    key={link.path}
                    to={link.path}
                    onClick={() => setIsMobileMenuOpen(false)}
                    className={`flex items-center px-4 py-3 mb-2 rounded-lg text-sm transition-colors ${
                      location.pathname === link.path
                        ? 'bg-blue-50 text-blue-700'
                        : 'text-gray-600 hover:bg-gray-100'
                    }`}
                  >
                    <span className="mr-3">{link.icon}</span>
                    {link.name}
                  </Link>
                ))}
                
                <button
                  onClick={() => {
                    handleLogout();
                    setIsMobileMenuOpen(false);
                  }}
                  className="flex items-center w-full px-4 py-3 mt-4 text-sm text-gray-600 rounded-lg hover:bg-gray-100 transition-colors"
                >
                  <LogOut size={20} className="mr-3" />
                  Sign Out
                </button>
              </nav>
            </motion.div>
          </div>
        )}
        
        {/* Main Content Area */}
        <main className="flex-1 overflow-y-auto bg-gray-50 p-4 sm:p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default MainLayout;