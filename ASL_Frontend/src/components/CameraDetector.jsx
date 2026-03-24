import React, { useRef, useEffect, useState, useCallback } from 'react';
import './CameraDetector.css';

// Standard MediaPipe Hand Connections
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8], // Index
  [5, 9], [9, 10], [10, 11], [11, 12], // Middle
  [9, 13], [13, 14], [14, 15], [15, 16], // Ring
  [13, 17], [17, 18], [18, 19], [19, 20], // Pinky
  [0, 17] // Palm Base
];

const CameraDetector = ({ targetSign, onCorrectSign }) => {
  const videoRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const [hasPermission, setHasPermission] = useState(false);
  const [errorMSG, setErrorMSG] = useState('');
  const [feedback, setFeedback] = useState([]);
  const [score, setScore] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Setup Camera
  useEffect(() => {
    let stream = null;

    const startCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ 
          video: { facingMode: 'user', width: 640, height: 480 } 
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setHasPermission(true);
      } catch (err) {
        console.error("Camera access denied:", err);
        setErrorMSG("Please allow camera access to use this feature.");
        setHasPermission(false);
      }
    };

    startCamera();

    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const drawTracking = (landmarksData) => {
    const canvas = overlayCanvasRef.current;
    if (!canvas || !videoRef.current) return;
    
    // Ensure display canvas matches video sizing exactly
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext('2d');
    
    ctx.clearRect(0, 0, canvas.width, canvas.height); // clear previous frame

    // Mirroring transformation because the video is mirrored via CSS
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    
    // Backend returns { right_hand: [{x, y, z}], left_hand: [] }
    const hands = [landmarksData.right_hand, landmarksData.left_hand].filter(h => h && h.length > 0);
    
    for (const hand of hands) {
      // Draw Bones
      ctx.lineWidth = 4;
      ctx.strokeStyle = "rgba(0, 255, 0, 0.7)";
      for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
        if (hand[startIdx] && hand[endIdx]) {
          const start = hand[startIdx];
          const end = hand[endIdx];
          ctx.beginPath();
          ctx.moveTo(start.x * canvas.width, start.y * canvas.height);
          ctx.lineTo(end.x * canvas.width, end.y * canvas.height);
          ctx.stroke();
        }
      }

      // Draw Joints
      ctx.fillStyle = "white";
      for (const landmark of hand) {
        ctx.beginPath();
        ctx.arc(landmark.x * canvas.width, landmark.y * canvas.height, 5, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
    
    ctx.restore();
  };

  // Frame Capture and Analysis Loop
  const analyzeFrame = useCallback(async () => {
    if (!videoRef.current || !captureCanvasRef.current || !hasPermission || isAnalyzing) return;
    if (videoRef.current.videoWidth === 0) return;

    setIsAnalyzing(true);
    const canvas = captureCanvasRef.current;
    const video = videoRef.current;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    
    // Draw current video frame to hidden canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Get base64 string
    const frameBase64 = canvas.toDataURL('image/jpeg', 0.8);

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          frame_base64: frameBase64,
          target_sign: targetSign,
          mode: 'static',
          include_landmarks: true // Ask for skeleton data
        })
      });

      if (!response.ok) throw new Error(`Server returned ${response.status}`);

      const result = await response.json();
      
      if (result.hand_detected) {
        setScore(result.overall_score || 0);
        setFeedback(result.messages || []);
        
        if (result.landmarks) {
          drawTracking(result.landmarks);
        }
        
        if (result.is_correct) {
          onCorrectSign(result.overall_score);
        }
      } else {
        setScore(0);
        setFeedback(["No hand detected. Please position your hand in frame."]);
        // Clear tracking overlay if no hand
        if (overlayCanvasRef.current) {
           const oc = overlayCanvasRef.current;
           oc.getContext('2d').clearRect(0, 0, oc.width, oc.height);
        }
      }
    } catch (err) {
      console.error("Analysis error:", err);
    } finally {
      setIsAnalyzing(false);
    }
  }, [hasPermission, targetSign, onCorrectSign, isAnalyzing]);

  // Run loop
  useEffect(() => {
    if (!hasPermission) return;
    const interval = setInterval(() => {
      analyzeFrame();
    }, 1000); // 1 FPS analysis for battery/server saving
    return () => clearInterval(interval);
  }, [analyzeFrame, hasPermission]);

  return (
    <div className="camera-detector-container">
      {errorMSG ? (
        <div className="camera-error">
          <p>⚠️ {errorMSG}</p>
        </div>
      ) : (
        <div className="camera-wrapper">
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            muted 
            className="webcam-video"
          />
          <canvas ref={captureCanvasRef} style={{ display: 'none' }} />
          
          <canvas 
            ref={overlayCanvasRef} 
            className="tracking-overlay" 
          />
          
          <div className="camera-overlay">
            <div className={`score-badge ${score >= 75 ? 'good' : 'bad'}`}>
              Score: {Math.round(score)}
            </div>
          </div>
          
          {feedback.length > 0 && (
            <div className="feedback-panel">
              <ul className="feedback-list">
                {feedback.map((msg, idx) => (
                  <li key={idx}>{msg}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default CameraDetector;

