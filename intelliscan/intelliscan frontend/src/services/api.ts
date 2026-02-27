import axios, { AxiosError } from "axios";

// Define the base URL based on environment
const API_BASE_URL = "http://127.0.0.1:5003";

// Create axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Define response types
export interface ClassificationResult {
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface PredictionResponse {
  classification: ClassificationResult;
  segmentation_mask_url: string;
}

// Define custom error messages
const getErrorMessage = (error: unknown): string => {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError;

    if (!axiosError.response) {
      return "Unable to connect to the server. Please check if the server is running and try again.";
    }

    switch (axiosError.response.status) {
      case 400:
        return "Invalid file format or corrupted image. Please try uploading a different file.";
      case 413:
        return "File size too large. Please upload a smaller file.";
      case 415:
        return "Unsupported file type. Please upload a valid image file.";
      case 500:
        return "Server error occurred while processing the image. Please try again later.";
      default:
        return `Upload failed: ${axiosError.response.status} - ${axiosError.response.statusText}`;
    }
  }
  return "An unexpected error occurred. Please try again.";
};

// Define API functions
// export const uploadMRIScan = async (
//   imageFile: File
// ): Promise<PredictionResponse> => {
//   try {
//     const formData = new FormData();
//     formData.append("image", imageFile);

//     const response = await apiClient.post<PredictionResponse>(
//       "/predict",
//       formData,
//       {
//         headers: {
//           "Content-Type": "multipart/form-data",
//         },
//         timeout: 30000, // 30 second timeout
//         validateStatus: (status) => status < 500, // Handle 4xx errors in try block
//       }
//     );

//     if (response.status !== 200) {
//       throw new Error(getErrorMessage({ response } as AxiosError));
//     }

//     const data = response.data;

//     // convert relative mask url → full backend url
//     // data.segmentation_mask_url = API_BASE_URL + data.segmentation_mask_url;
//     if (!data.segmentation_mask_url.startsWith("http")) {
//       data.segmentation_mask_url = API_BASE_URL + data.segmentation_mask_url;
//     }

//     return data;
//     // if (response.status !== 200) {
//     //   throw new Error(getErrorMessage({ response } as AxiosError));
//     // }

//     // return response.data;
//   } catch (error) {
//     console.error("Error uploading MRI scan:", error);
//     throw new Error(getErrorMessage(error));
//   }
// };
export const uploadMRIScan = async (
  imageFile: File
): Promise<PredictionResponse> => {
  try {
    const formData = new FormData();
    formData.append("image", imageFile);

    const response = await apiClient.post<PredictionResponse>(
      "/predict",
      formData,
      {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 30000,
      }
    );

    if (response.status !== 200) {
      throw new Error("Upload failed");
    }

    const data = response.data;

    // force correct mask URL always
    data.segmentation_mask_url = "http://127.0.0.1:5003/mask";

    return data;

    // const data = response.data;

    // // FIX — always absolute URL
    // if (!data.segmentation_mask_url.startsWith("http")) {
    //   data.segmentation_mask_url = API_BASE_URL + data.segmentation_mask_url;
    // }

    // console.log("MASK URL FROM BACKEND:", data.segmentation_mask_url);

    // return data;
  } catch (error) {
    console.error(error);
    throw error;
  }
};

// Define report types
export interface Report {
  id: string;
  patientId: string;
  patientName: string;
  patientDetails: {
    age?: number;
    gender?: string;
    contactNumber?: string;
    email: string;
    registrationDate: string;
  };
  createdAt: string;
  classification: ClassificationResult;
  maskImageUrl: string;
  status: "pending" | "reviewed";
  doctorNotes?: string;
}

// Mock data store
let MOCK_REPORTS: Report[] = [];

export const getPatientReports = async (
  patientId: string
): Promise<Report[]> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      const reports = MOCK_REPORTS.filter(
        (report) => report.patientId === patientId
      );
      resolve(reports);
    }, 1000);
  });
};

export const getDoctorReports = async (): Promise<Report[]> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(MOCK_REPORTS);
    }, 1000);
  });
};

export const getReportById = async (
  reportId: string
): Promise<Report | null> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      const report = MOCK_REPORTS.find((r) => r.id === reportId) || null;
      resolve(report);
    }, 1000);
  });
};

export const addDoctorNotes = async (
  reportId: string,
  notes: string
): Promise<Report> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      const reportIndex = MOCK_REPORTS.findIndex((r) => r.id === reportId);
      if (reportIndex !== -1) {
        MOCK_REPORTS[reportIndex] = {
          ...MOCK_REPORTS[reportIndex],
          doctorNotes: notes,
          status: "reviewed",
        };
        resolve(MOCK_REPORTS[reportIndex]);
      } else {
        throw new Error("Report not found");
      }
    }, 1000);
  });
};

export const createReport = async (
  patientId: string,
  patientName: string,
  patientDetails: Report["patientDetails"],
  classification: ClassificationResult,
  maskImageUrl: string
): Promise<Report> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      const newReport: Report = {
        id: `r${Date.now()}`,
        patientId,
        patientName,
        patientDetails,
        createdAt: new Date().toISOString(),
        classification,
        maskImageUrl,
        status: "pending",
      };

      MOCK_REPORTS.push(newReport);
      resolve(newReport);
    }, 1000);
  });
};
