import React, { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { toast } from "react-toastify";
import { useAuth } from "../../contexts/AuthContext";
import {
  uploadMRIScan,
  createReport,
  ClassificationResult,
} from "../../services/api";
import {
  UploadCloud,
  FileType,
  CheckCircle,
  AlertTriangle,
  RotateCw,
} from "lucide-react";

const UploadScan: React.FC = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [filePreview, setFilePreview] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadComplete, setUploadComplete] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const progressIntervalRef = useRef<NodeJS.Timeout>();

  // Cleanup function for the progress interval
  const cleanupProgressInterval = () => {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = undefined;
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];

      // Check file type
      if (!selectedFile.type.startsWith("image/")) {
        toast.error("Please select a valid image file");
        return;
      }

      // Check file size (5MB max)
      if (selectedFile.size > 5 * 1024 * 1024) {
        toast.error("File size must be less than 5MB");
        return;
      }

      setFile(selectedFile);

      // Create and set preview URL
      const reader = new FileReader();
      reader.onload = (e) => {
        setFilePreview(e.target?.result as string);
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];

      // Check file type
      if (!droppedFile.type.startsWith("image/")) {
        toast.error("Please select a valid image file");
        return;
      }

      // Check file size (5MB max)
      if (droppedFile.size > 5 * 1024 * 1024) {
        toast.error("File size must be less than 5MB");
        return;
      }

      setFile(droppedFile);

      // Create and set preview URL
      const reader = new FileReader();
      reader.onload = (e) => {
        setFilePreview(e.target?.result as string);
      };
      reader.readAsDataURL(droppedFile);
    }
  };

  // const handleUpload = async () => {
  //   if (!file || !user) return;

  //   setIsUploading(true);
  //   setUploadProgress(0);

  //   // Simulate upload progress
  //   progressIntervalRef.current = setInterval(() => {
  //     setUploadProgress((prev) => {
  //       if (prev >= 90) {
  //         cleanupProgressInterval();
  //         return 90;
  //       }
  //       return prev + 10;
  //     });
  //   }, 300);

  //   try {
  //     // Call API to upload and analyze the scan
  //     const result = await uploadMRIScan(file);

  //     // Complete the progress bar
  //     cleanupProgressInterval();
  //     setUploadProgress(100);
  //     setUploadComplete(true);

  //     // Create a new report with patient details
  //     const maskImageUrl = result.segmentation_mask_url;
  //     const patientDetails = {
  //       age: user.age,
  //       gender: user.gender,
  //       contactNumber: user.contactNumber,
  //       email: user.email,
  //       registrationDate: user.registrationDate,
  //     };

  //     const report = await createReport(
  //       user.id,
  //       user.name,
  //       patientDetails,
  //       result.classification,
  //       maskImageUrl
  //     );

  //     toast.success("MRI scan uploaded and analyzed successfully");

  //     // Navigate to the report page
  //     setTimeout(() => {
  //       navigate(`/patient/reports/${report.id}`);
  //     }, 1500);
  //   } catch (error) {
  //     cleanupProgressInterval();
  //     console.error("Upload error:", error);
  //     toast.error(
  //       error instanceof Error
  //         ? error.message
  //         : "Failed to upload and analyze the scan. Please try again."
  //     );
  //     setIsUploading(false);
  //     setUploadProgress(0);
  //   }
  // };
  const handleUpload = async () => {
    if (!file || !user) return;

    setIsUploading(true);
    setUploadProgress(0);

    // Simulate upload progress
    progressIntervalRef.current = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 90) {
          cleanupProgressInterval();
          return 90;
        }
        return prev + 10;
      });
    }, 300);

    try {
      // Upload + AI prediction
      const result = await uploadMRIScan(file);

      cleanupProgressInterval();
      setUploadProgress(100);
      setUploadComplete(true);

      // ⭐ IMPORTANT FIX (DO NOT CHANGE THIS)
      const maskImageUrl = result.segmentation_mask_url;

      console.log("MASK URL SAVED:", maskImageUrl);

      const patientDetails = {
        age: user.age,
        gender: user.gender,
        contactNumber: user.contactNumber,
        email: user.email,
        registrationDate: user.registrationDate,
      };

      const report = await createReport(
        user.id,
        user.name,
        patientDetails,
        result.classification,
        maskImageUrl
      );

      toast.success("MRI scan uploaded and analyzed successfully");

      setTimeout(() => {
        navigate(`/patient/reports/${report.id}`);
      }, 1500);
    } catch (error) {
      cleanupProgressInterval();
      console.error("Upload error:", error);
      toast.error(
        error instanceof Error
          ? error.message
          : "Failed to upload and analyze the scan."
      );
      setIsUploading(false);
      setUploadProgress(0);
    }
  };
  const resetUpload = () => {
    setFile(null);
    setFilePreview(null);
    setIsUploading(false);
    setUploadProgress(0);
    setUploadComplete(false);
    cleanupProgressInterval();
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Cleanup on component unmount
  React.useEffect(() => {
    return () => {
      cleanupProgressInterval();
    };
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Upload MRI Scan</h1>
        <p className="text-gray-600 mt-1">
          Upload your MRI scan image for brain tumor analysis
        </p>
      </div>

      <div className="bg-white rounded-xl shadow-md overflow-hidden">
        <div className="p-6 border-b border-gray-200">
          <h2 className="text-lg font-medium text-gray-800">Upload Image</h2>
          <p className="text-sm text-gray-500 mt-1">
            Supported formats: JPG, PNG, JPEG. Maximum file size: 5MB.
          </p>
        </div>

        <div className="p-6">
          {/* Upload Area */}
          {!file ? (
            <div
              className="border-2 border-dashed border-gray-300 rounded-lg p-10 text-center cursor-pointer hover:bg-gray-50 transition-colors"
              onClick={() => fileInputRef.current?.click()}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="image/*"
                className="hidden"
              />

              <UploadCloud className="mx-auto h-12 w-12 text-gray-400" />
              <p className="mt-2 text-sm font-medium text-gray-900">
                Drag and drop your MRI scan image here
              </p>
              <p className="mt-1 text-xs text-gray-500">or</p>
              <button
                type="button"
                className="mt-2 inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                Browse Files
              </button>
            </div>
          ) : (
            <div className="space-y-6">
              {/* File Preview */}
              <div className="bg-gray-50 rounded-lg p-6">
                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0">
                    <div className="h-12 w-12 rounded-lg bg-blue-100 flex items-center justify-center">
                      <FileType className="h-6 w-6 text-blue-600" />
                    </div>
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {file.name}
                    </p>
                    <p className="text-sm text-gray-500">
                      {(file.size / 1024 / 1024).toFixed(2)} MB • {file.type}
                    </p>
                  </div>
                  <div className="flex-shrink-0">
                    <button
                      type="button"
                      onClick={resetUpload}
                      disabled={isUploading}
                      className="inline-flex items-center px-3 py-1 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                    >
                      Change
                    </button>
                  </div>
                </div>

                {filePreview && (
                  <div className="mt-4 relative rounded-lg overflow-hidden border border-gray-200">
                    <img
                      src={filePreview}
                      alt="MRI scan preview"
                      className="w-full h-64 object-contain"
                    />
                  </div>
                )}
              </div>

              {/* Upload Progress */}
              {isUploading && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="text-sm font-medium text-gray-700">
                      {uploadComplete ? "Analysis complete!" : "Processing..."}
                    </div>
                    <div className="text-sm font-medium text-gray-700">
                      {uploadProgress}%
                    </div>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div
                      className={`h-2.5 rounded-full ${
                        uploadComplete ? "bg-green-500" : "bg-blue-500"
                      }`}
                      style={{ width: `${uploadProgress}%` }}
                    ></div>
                  </div>
                  <div className="text-sm text-gray-500">
                    {uploadComplete ? (
                      <div className="flex items-center text-green-600">
                        <CheckCircle className="h-4 w-4 mr-1" />
                        Analysis complete. Redirecting to results...
                      </div>
                    ) : (
                      <div className="flex items-center">
                        <RotateCw className="h-4 w-4 mr-1 animate-spin" />
                        Analyzing MRI scan. Please wait...
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              {!isUploading && (
                <div className="flex justify-end space-x-3">
                  <button
                    type="button"
                    onClick={resetUpload}
                    className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={handleUpload}
                    className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                  >
                    <UploadCloud className="h-4 w-4 mr-2" />
                    Upload and Analyze
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Guidelines Section */}
      <div className="bg-blue-50 rounded-xl shadow-sm p-6 mt-8">
        <h2 className="text-lg font-medium text-gray-800 mb-4">
          MRI Scan Guidelines
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-medium text-gray-800 mb-2 flex items-center">
              <CheckCircle className="h-5 w-5 text-green-500 mr-2" />
              Recommended
            </h3>
            <ul className="space-y-2 text-sm text-gray-600">
              <li className="flex items-start">
                <span className="text-green-500 mr-2">•</span>
                Clear, high-resolution images
              </li>
              <li className="flex items-start">
                <span className="text-green-500 mr-2">•</span>
                Standard MRI formats: T1, T2, FLAIR sequences
              </li>
              <li className="flex items-start">
                <span className="text-green-500 mr-2">•</span>
                Proper orientation (axial, sagittal, or coronal)
              </li>
              <li className="flex items-start">
                <span className="text-green-500 mr-2">•</span>
                Complete brain scans showing all regions
              </li>
            </ul>
          </div>

          <div>
            <h3 className="font-medium text-gray-800 mb-2 flex items-center">
              <AlertTriangle className="h-5 w-5 text-yellow-500 mr-2" />
              Not Recommended
            </h3>
            <ul className="space-y-2 text-sm text-gray-600">
              <li className="flex items-start">
                <span className="text-red-500 mr-2">•</span>
                Blurry or low-resolution images
              </li>
              <li className="flex items-start">
                <span className="text-red-500 mr-2">•</span>
                Images with artifacts or excessive noise
              </li>
              <li className="flex items-start">
                <span className="text-red-500 mr-2">•</span>
                Partial brain scans missing important regions
              </li>
              <li className="flex items-start">
                <span className="text-red-500 mr-2">•</span>
                MRI scans with overlay text or annotations
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-4 pt-4 border-t border-blue-100">
          <p className="text-sm text-gray-600">
            <strong>Note:</strong> For optimal analysis results, please ensure
            your MRI scans are recent (within the last 6 months) and follow the
            recommended guidelines. The AI analysis works best with standard
            clinical MRI protocols.
          </p>
        </div>
      </div>
    </motion.div>
  );
};

export default UploadScan;
