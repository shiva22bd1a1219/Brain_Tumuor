import React from "react";
import { motion } from "framer-motion";
import { Dna, AlertTriangle, BarChart3, FileCheck } from "lucide-react";
import { ClassificationResult } from "../services/api";
import { Doughnut } from "react-chartjs-2";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
} from "chart.js";

// Register Chart.js components
ChartJS.register(
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title
);

interface TumorAnalysisResultProps {
  classification: ClassificationResult;
  maskImageUrl: string;
}

const TumorAnalysisResult: React.FC<TumorAnalysisResultProps> = ({
  classification,
  maskImageUrl,
}) => {
  const tumorTypeDescriptions = {
    glioma:
      "Gliomas originate in the glial cells that surround and support neurons in the brain. They can be slow-growing or aggressive.",
    meningioma:
      "Meningiomas develop in the meninges, the membranes that surround the brain and spinal cord. Most are benign.",
    pituitary:
      "Pituitary tumors form in the pituitary gland and can affect hormone production.",
    no_tumor: "No tumor was detected in the scan.",
  };

  const chartData = {
    labels: Object.keys(classification.probabilities),
    datasets: [
      {
        data: Object.values(classification.probabilities),
        backgroundColor: [
          "rgba(255, 99, 132, 0.7)",
          "rgba(54, 162, 235, 0.7)",
          "rgba(255, 206, 86, 0.7)",
          "rgba(75, 192, 192, 0.7)",
        ],
        borderColor: [
          "rgba(255, 99, 132, 1)",
          "rgba(54, 162, 235, 1)",
          "rgba(255, 206, 86, 1)",
          "rgba(75, 192, 192, 1)",
        ],
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: "bottom" as const,
      },
      tooltip: {
        callbacks: {
          label: function (context: any) {
            const label = context.label || "";
            const value = context.raw;
            return `${label}: ${(value * 100).toFixed(1)}%`;
          },
        },
      },
    },
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return "text-green-600";
    if (confidence >= 0.7) return "text-blue-600";
    if (confidence >= 0.5) return "text-yellow-600";
    return "text-red-600";
  };

  const confidenceColor = getConfidenceColor(classification.confidence);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="bg-white rounded-xl shadow-md overflow-hidden mt-6"
    >
      <div className="p-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
          <Dna className="w-5 h-5 mr-2 text-blue-500" />
          MRI Scan Analysis Results
        </h2>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* LEFT SIDE */}
          <div className="space-y-4">
            <div className="bg-blue-50 p-4 rounded-lg">
              <h3 className="text-lg font-medium text-gray-800 mb-2 flex items-center">
                <FileCheck className="w-4 h-4 mr-2 text-blue-600" />
                Classification Result
              </h3>

              <div className="flex items-center mb-3">
                <div className="text-3xl font-bold text-gray-900 mr-2">
                  {classification.label}
                </div>
                <span className={`text-sm font-medium ${confidenceColor}`}>
                  ({(classification.confidence * 100).toFixed(1)}% confidence)
                </span>
              </div>

              <p className="text-sm text-gray-600">
                {tumorTypeDescriptions[
                  classification.label as keyof typeof tumorTypeDescriptions
                ] || "This is a detected tumor type."}
              </p>

              {classification.confidence < 0.7 && (
                <div className="mt-3 flex items-start bg-yellow-50 p-3 rounded-md border border-yellow-200">
                  <AlertTriangle className="w-5 h-5 text-yellow-500 mr-2 flex-shrink-0 mt-0.5" />
                  <p className="text-sm text-yellow-700">
                    The confidence level for this prediction is relatively low.
                    Consider consulting with a specialist.
                  </p>
                </div>
              )}
            </div>

            <div>
              <h3 className="text-lg font-medium text-gray-800 mb-3 flex items-center">
                <BarChart3 className="w-4 h-4 mr-2 text-blue-600" />
                Classification Probabilities
              </h3>

              <div className="h-64 w-full">
                <Doughnut data={chartData} options={chartOptions} />
              </div>
            </div>
          </div>

          {/* RIGHT SIDE - SEGMENTATION */}
          <div>
            <h3 className="text-lg font-medium text-gray-800 mb-3">
              Tumor Segmentation
            </h3>

            <div className="h-64 w-full rounded-lg overflow-hidden border border-gray-200 flex items-center justify-center bg-black">
              <img
                src={`${maskImageUrl}?t=${Date.now()}`}
                alt="Tumor segmentation"
                className="h-full w-full object-contain"
                onError={() => {
                  console.log("MASK FAILED:", maskImageUrl);
                }}
              />
            </div>

            <p className="mt-2 text-sm text-gray-600">
              The green overlay shows the segmented tumor region identified in
              the MRI scan.
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default TumorAnalysisResult;
// import React from "react";
// import { motion } from "framer-motion";
// import { Dna, AlertTriangle, BarChart3, FileCheck } from "lucide-react";
// import { ClassificationResult } from "../services/api";
// import { Doughnut } from "react-chartjs-2";
// import {
//   Chart as ChartJS,
//   ArcElement,
//   Tooltip,
//   Legend,
//   CategoryScale,
//   LinearScale,
//   PointElement,
//   LineElement,
//   Title,
// } from "chart.js";

// // Register Chart.js components
// ChartJS.register(
//   ArcElement,
//   Tooltip,
//   Legend,
//   CategoryScale,
//   LinearScale,
//   PointElement,
//   LineElement,
//   Title
// );

// interface TumorAnalysisResultProps {
//   classification: ClassificationResult;
//   maskImageUrl: string;
// }

// const TumorAnalysisResult: React.FC<TumorAnalysisResultProps> = ({
//   classification,
//   maskImageUrl,
// }) => {
//   const tumorTypeDescriptions = {
//     glioma:
//       "Gliomas originate in the glial cells that surround and support neurons in the brain. They can be slow-growing or aggressive.",
//     meningioma:
//       "Meningiomas develop in the meninges, the membranes that surround the brain and spinal cord. Most are benign.",
//     pituitary:
//       "Pituitary tumors form in the pituitary gland and can affect hormone production.",
//     no_tumor: "No tumor was detected in the scan.",
//   };

//   const chartData = {
//     labels: Object.keys(classification.probabilities),
//     datasets: [
//       {
//         data: Object.values(classification.probabilities),
//         backgroundColor: [
//           "rgba(255, 99, 132, 0.7)",
//           "rgba(54, 162, 235, 0.7)",
//           "rgba(255, 206, 86, 0.7)",
//           "rgba(75, 192, 192, 0.7)",
//         ],
//         borderColor: [
//           "rgba(255, 99, 132, 1)",
//           "rgba(54, 162, 235, 1)",
//           "rgba(255, 206, 86, 1)",
//           "rgba(75, 192, 192, 1)",
//         ],
//         borderWidth: 1,
//       },
//     ],
//   };

//   const chartOptions = {
//     responsive: true,
//     plugins: {
//       legend: {
//         position: "bottom" as const,
//       },
//       tooltip: {
//         callbacks: {
//           label: function (context: any) {
//             const label = context.label || "";
//             const value = context.raw;
//             return `${label}: ${(value * 100).toFixed(1)}%`;
//           },
//         },
//       },
//     },
//   };

//   // Choose the right color based on confidence
//   const getConfidenceColor = (confidence: number) => {
//     if (confidence >= 0.9) return "text-green-600";
//     if (confidence >= 0.7) return "text-blue-600";
//     if (confidence >= 0.5) return "text-yellow-600";
//     return "text-red-600";
//   };

//   const confidenceColor = getConfidenceColor(classification.confidence);

//   return (
//     <motion.div
//       initial={{ opacity: 0, y: 20 }}
//       animate={{ opacity: 1, y: 0 }}
//       transition={{ duration: 0.5, delay: 0.2 }}
//       className="bg-white rounded-xl shadow-md overflow-hidden mt-6"
//     >
//       <div className="p-6">
//         <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
//           <Dna className="w-5 h-5 mr-2 text-blue-500" />
//           MRI Scan Analysis Results
//         </h2>

//         <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
//           {/* Tumor Classification */}
//           <div className="space-y-4">
//             <div className="bg-blue-50 p-4 rounded-lg">
//               <h3 className="text-lg font-medium text-gray-800 mb-2 flex items-center">
//                 <FileCheck className="w-4 h-4 mr-2 text-blue-600" />
//                 Classification Result
//               </h3>

//               <div className="flex items-center mb-3">
//                 <div className="text-3xl font-bold text-gray-900 mr-2">
//                   {classification.label}
//                 </div>
//                 <span className={`text-sm font-medium ${confidenceColor}`}>
//                   ({(classification.confidence * 100).toFixed(1)}% confidence)
//                 </span>
//               </div>

//               <p className="text-sm text-gray-600">
//                 {tumorTypeDescriptions[
//                   classification.label as keyof typeof tumorTypeDescriptions
//                 ] || "This is a detected tumor type."}
//               </p>

//               {classification.confidence < 0.7 && (
//                 <div className="mt-3 flex items-start bg-yellow-50 p-3 rounded-md border border-yellow-200">
//                   <AlertTriangle className="w-5 h-5 text-yellow-500 mr-2 flex-shrink-0 mt-0.5" />
//                   <p className="text-sm text-yellow-700">
//                     The confidence level for this prediction is relatively low.
//                     Consider consulting with a specialist.
//                   </p>
//                 </div>
//               )}
//             </div>

//             <div>
//               <h3 className="text-lg font-medium text-gray-800 mb-3 flex items-center">
//                 <BarChart3 className="w-4 h-4 mr-2 text-blue-600" />
//                 Classification Probabilities
//               </h3>

//               <div className="h-64 w-full">
//                 <Doughnut data={chartData} options={chartOptions} />
//               </div>
//             </div>
//           </div>

//           {/* Segmentation Mask */}
//                   <div>
//           <h3 className="text-lg font-medium text-gray-800 mb-3">
//             Tumor Segmentation
//           </h3>

//           <div className="h-64 w-full rounded-lg overflow-hidden border border-gray-200 flex items-center justify-center bg-black">
//             <img
//               src={`${maskImageUrl}?t=${Date.now()}`}
//               alt="Tumor segmentation"
//               className="h-full w-full object-contain"
//             />
//           </div>

//           <p className="mt-2 text-sm text-gray-600">
//             The green overlay shows the segmented tumor region identified in the MRI scan.
//           </p>
//         </div>
//           {/* <div>
//             <h3 className="text-lg font-medium text-gray-800 mb-3">
//               Tumor Segmentation
//             </h3>
//             <div className="rounded-lg overflow-hidden border border-gray-200">
//               <img
//                 src={`${maskImageUrl}?t=${Date.now()}`}
//                 alt="Tumor segmentation"
//                 className="w-full h-auto object-cover"
//                 onError={(e) => {
//                   console.log("MASK FAILED:", maskImageUrl);
//                 }}
//               />

//               {/* <img
//                 src={maskImageUrl}
//                 alt="Tumor segmentation"
//                 className="w-full h-auto object-cover"
//               /> */}
//             </div>
//             <p className="mt-2 text-sm text-gray-600">
//               The green overlay shows the segmented tumor region identified in
//               the MRI scan.
//             </p>
//           </div> */}
//         </div>
//       </div>
//     </motion.div>
//   );
// };

// export default TumorAnalysisResult;
