import React, { createContext, useContext, useState, useEffect } from 'react';

// Define user type
export type UserType = {
  id: string;
  name: string;
  email: string;
  role: 'patient' | 'doctor';
  age?: number;
  gender?: string;
  specialization?: string; // for doctors
  contactNumber?: string;
  registrationDate: string;
};

// Mock users for demonstration
const MOCK_USERS = [
  {
    id: 'p1',
    name: 'John Doe',
    email: 'patient@example.com',
    password: 'password',
    role: 'patient',
    age: 45,
    gender: 'Male',
    contactNumber: '+1 (555) 123-4567',
    registrationDate: '2023-01-15',
  },
  {
    id: 'd1',
    name: 'Dr. Jane Smith',
    email: 'doctor@example.com',
    password: 'password',
    role: 'doctor',
    specialization: 'Neurology',
    contactNumber: '+1 (555) 987-6543',
    registrationDate: '2022-11-20',
  }
];

interface AuthContextType {
  user: UserType | null;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<UserType>;
  register: (userData: any) => Promise<UserType>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<UserType | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  
  // Check for existing session on load
  useEffect(() => {
    const storedUser = localStorage.getItem('user');
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
    setIsLoading(false);
  }, []);
  
  const login = async (email: string, password: string): Promise<UserType> => {
    // In a real app, this would be an API call
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        const foundUser = MOCK_USERS.find(
          (u) => u.email === email && u.password === password
        );
        
        if (foundUser) {
          // Omit password before storing/returning
          const { password, ...userWithoutPassword } = foundUser;
          setUser(userWithoutPassword as UserType);
          localStorage.setItem('user', JSON.stringify(userWithoutPassword));
          resolve(userWithoutPassword as UserType);
        } else {
          reject(new Error('Invalid email or password'));
        }
      }, 1000);
    });
  };
  
  const register = async (userData: any): Promise<UserType> => {
    // In a real app, this would be an API call
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        // Check if email already exists
        const userExists = MOCK_USERS.some((u) => u.email === userData.email);
        
        if (userExists) {
          reject(new Error('Email already registered'));
          return;
        }
        
        // Create new user
        const newUser = {
          id: `p${Date.now()}`,
          ...userData,
          registrationDate: new Date().toISOString().split('T')[0],
        };
        
        // In a real app, would save to a database
        MOCK_USERS.push(newUser);
        
        // Omit password before storing/returning
        const { password, ...userWithoutPassword } = newUser;
        setUser(userWithoutPassword as UserType);
        localStorage.setItem('user', JSON.stringify(userWithoutPassword));
        resolve(userWithoutPassword as UserType);
      }, 1000);
    });
  };
  
  const logout = () => {
    setUser(null);
    localStorage.removeItem('user');
  };
  
  return (
    <AuthContext.Provider value={{ user, isLoading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};