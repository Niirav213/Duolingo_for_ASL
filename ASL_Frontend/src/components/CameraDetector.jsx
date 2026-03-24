import React, { useRef, useEffect, useState, useCallback } from 'react';
import './CameraDetector.css';

// 1. PLACE THIS AT THE TOP (AFTER IMPORTS)
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], 
  [0, 5], [5, 6], [6, 7], [7, 8], 
  [0, 9], [9, 10], [11, 12], 
  [0, 13], [13, 14], [15, 16], 
  [0, 17], [17, 18], [19, 20], 
  [5, 9], [9, 13], [13, 17]
];

const JOINT_MAP = {
  2: "thumb_mcp", 3: "thumb_ip", 6: "index_pip", 5: "index_mcp",
  10: "middle_pip", 9: "middle_mcp", 14: "ring_pip", 13: "ring_mcp",
  18: "pinky_pip", 17: "pinky_mcp"
};

const COLOR_VALS = { green: "#00d250", orange: "#ffa500", red: "#dc3200" };

const CameraDetector = ({ targetSign, onCorrectSign }) => {
  const videoRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const [hasPermission, setHasPermission] = useState(false);
  const [errorMSG, setErrorMSG] = useState('');
  const [feedback, setFeedback] = useState([]);
  const [score, setScore] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [detectedSign, setDetectedSign] = useState(null);
  const isAnalyzingRef = useRef(false);
  const matchCountRef = useRef(0);  // Track consecutive correct matches

  useEffect(() => {
    let stream = null;
    const startCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ 
          video: { facingMode: 'user', width: 640, height: 480 } 
        });
        if (videoRef.current) videoRef.current.srcObject = stream;
        setHasPermission(true);
      } catch (err) {
        setErrorMSG("Please allow camera access to use this feature.");
        setHasPermission(false);
      }
    };
    startCamera();
    return () => stream?.getTracks().forEach(track => track.stop());
  }, []);

  // 2. REPLACE YOUR OLD drawTracking WITH THIS
const drawTracking = useCallback((result) => {
  const canvas = overlayCanvasRef.current;
  const video = videoRef.current;
  if (!canvas || !video || !result.landmarks) return;

  const ctx = canvas.getContext('2d');
  const w = video.videoWidth;
  const h = video.videoHeight;
  
  canvas.width = w;
  canvas.height = h;
  ctx.clearRect(0, 0, w, h);

  ctx.save();
  // Mirror the drawing to match the webcam CSS
  ctx.translate(w, 0);
  ctx.scale(-1, 1);

  // Draw Skeleton Connections (Gray lines)
  ctx.strokeStyle = "rgba(160, 160, 160, 0.6)";
  ctx.lineWidth = 2;
  HAND_CONNECTIONS.forEach(([a, b]) => {
    const start = result.landmarks[a];
    const end = result.landmarks[b];
    if (start && end) {
      ctx.beginPath();
      ctx.moveTo(start.x * w, start.y * h);
      ctx.lineTo(end.x * w, end.y * h);
      ctx.stroke();
    }
  });

  // Draw Colored Joints
  result.landmarks.forEach((lm, idx) => {
    const jointName = JOINT_MAP[idx];
    const colorKey = result.joint_colors ? result.joint_colors[jointName] : null;
    const color = COLOR_VALS[colorKey] || "#ffffff";

    ctx.beginPath();
    ctx.arc(lm.x * w, lm.y * h, jointName ? 7 : 4, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = "black";
    ctx.lineWidth = 1;
    ctx.stroke();
  });

  ctx.restore();
}, []);

  // 3. Frame analysis function
  const analyzeFrame = useCallback(async () => {
  if (!videoRef.current || isAnalyzingRef.current) return;
  const video = videoRef.current;
  if (video.videoWidth === 0) return;

  isAnalyzingRef.current = true;
  setIsAnalyzing(true);
  const canvas = captureCanvasRef.current;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const frameBase64 = canvas.toDataURL('image/jpeg', 0.5);

  try {
    // Connect to the CV-Module
    const response = await fetch('/analyze', { 
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        frame_base64: frameBase64,
        target_sign: targetSign,
        mode: 'static',
        include_landmarks: true 
      })
    });

    const result = await response.json();
    console.log('[CV Response]', { 
      hand: result.hand_detected, 
      sign: result.detected_sign, 
      conf: result.confidence,
      score: result.overall_score, 
      correct: result.is_correct,
      target: targetSign 
    });
    
    if (result.hand_detected && result.landmarks) {
      setScore(result.overall_score || 0);
      setFeedback(result.messages || []);
      setDetectedSign(result.detected_sign || null);
      drawTracking(result); // This draws the skeleton
      
      // Check if sign matches - use both is_correct flag AND manual sign comparison
      const signMatchesTarget = result.detected_sign && 
        result.detected_sign.toUpperCase() === targetSign.toUpperCase();
      
      if (result.is_correct || (signMatchesTarget && result.confidence >= 0.35)) {
        matchCountRef.current += 1;
        console.log(`[MATCH] Count: ${matchCountRef.current}`);
        // Require 2 consecutive matches for reliability
        if (matchCountRef.current >= 2) {
          const finalScore = Math.max(result.overall_score || 0, result.confidence * 100 || 60);
          console.log(`[CORRECT SIGN TRIGGERED] Score: ${finalScore}`);
          onCorrectSign(finalScore);
          matchCountRef.current = 0;
        }
      } else {
        matchCountRef.current = 0;
      }
    } else {
      setScore(0);
      setDetectedSign(null);
      setFeedback([]);
      matchCountRef.current = 0;
      const oc = overlayCanvasRef.current;
      if (oc) oc.getContext('2d').clearRect(0, 0, oc.width, oc.height);
    }
  } catch (err) {
    console.error("CV analysis error:", err.message);
  } finally {
    isAnalyzingRef.current = false;
    setIsAnalyzing(false);
  }
}, [targetSign, onCorrectSign, drawTracking]);

  // 4. CONTINUOUS ANALYSIS LOOP — runs every 500ms
  useEffect(() => {
    if (!hasPermission) return;
    const intervalId = setInterval(() => {
      analyzeFrame();
    }, 500);
    return () => clearInterval(intervalId);
  }, [hasPermission, analyzeFrame]);

  return (
    <div className="camera-detector-container">
      {errorMSG ? (
        <div className="camera-error"><p>⚠️ {errorMSG}</p></div>
      ) : (
        <div className="camera-wrapper">
          <video ref={videoRef} autoPlay playsInline muted className="webcam-video" />
          <canvas ref={captureCanvasRef} style={{ display: 'none' }} />
          <canvas ref={overlayCanvasRef} className="tracking-overlay" />
          
          <div className="camera-overlay">
            <div className={`score-badge ${score >= 75 ? 'good' : 'bad'}`}>
              Score: {Math.round(score)}
            </div>
            {detectedSign && (
              <div style={{
                position: 'absolute', top: '12px', left: '12px',
                background: detectedSign === targetSign ? '#00d250' : '#ff6b35',
                color: 'white', padding: '8px 16px', borderRadius: '12px',
                fontWeight: 'bold', fontSize: '1.1rem',
                boxShadow: '0 2px 8px rgba(0,0,0,0.3)'
              }}>
                Detected: {detectedSign}
              </div>
            )}
            {isAnalyzing && (
              <div style={{
                position: 'absolute', bottom: '12px', left: '12px',
                background: 'rgba(0,0,0,0.6)', color: '#4af', padding: '4px 10px',
                borderRadius: '8px', fontSize: '0.8rem'
              }}>⏳ Analyzing...</div>
            )}
          </div>
          
          {feedback.length > 0 && (
            <div className="feedback-panel">
              <ul className="feedback-list">
                {feedback.map((msg, idx) => <li key={idx}>{msg}</li>)}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default CameraDetector;