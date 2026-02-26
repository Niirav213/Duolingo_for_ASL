import React, { useEffect, useRef, useState } from 'react';
import { useGameStore } from '../store';

export const GestureCamera = ({ expectedSign, onDetection, lessonId }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [detectedSign, setDetectedSign] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    // Start video stream
    const startVideo = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user' },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error('Error accessing camera:', err);
      }
    };

    startVideo();

    // Connect WebSocket for gesture detection
    const token = localStorage.getItem('accessToken');
    const wsUrl = `ws://localhost:8000/api/v1/ws/gesture/${token}`;

    try {
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
      };

      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.status === 'success') {
          setDetectedSign(data.detected_sign);
          setConfidence(data.confidence);
          onDetection({
            sign: data.detected_sign,
            confidence: data.confidence,
            correct: data.correct,
          });
        }
      };

      wsRef.current.onerror = () => {
        console.error('WebSocket error');
        setIsConnected(false);
      };

      wsRef.current.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
      };
    } catch (err) {
      console.error('WebSocket connection failed:', err);
    }

    return () => {
      if (videoRef.current?.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach((track) => track.stop());
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [onDetection]);

  const captureAndSend = () => {
    if (!canvasRef.current || !videoRef.current) return;

    const ctx = canvasRef.current.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, 224, 224);

    canvasRef.current.toBlob((blob) => {
      const reader = new FileReader();
      reader.onload = () => {
        const base64Data = reader.result.split(',')[1];

        if (wsRef.current && isConnected) {
          wsRef.current.send(
            JSON.stringify({
              image_data: base64Data,
              lesson_id: lessonId,
              expected_sign: expectedSign,
            })
          );
        }
      };
      reader.readAsDataURL(blob);
    });
  };

  useEffect(() => {
    const interval = setInterval(captureAndSend, 500); // Send frame every 500ms
    return () => clearInterval(interval);
  }, [expectedSign, lessonId, isConnected]);

  return (
    <div className="gesture-camera bg-gray-900 rounded-lg overflow-hidden shadow-lg">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        className="w-full h-96 object-cover mirror"
      />
      <canvas ref={canvasRef} width={224} height={224} className="hidden" />

      <div className="p-4 bg-gray-800">
        <div className="flex justify-between items-center mb-3">
          <div>
            <p className="text-gray-400 text-sm">Expected Sign</p>
            <p className="text-white text-2xl font-bold">{expectedSign}</p>
          </div>
          <div>
            <p className="text-gray-400 text-sm">Detected</p>
            <p className="text-green-400 text-2xl font-bold">{detectedSign || '--'}</p>
          </div>
          <div>
            <p className="text-gray-400 text-sm">Confidence</p>
            <p className="text-yellow-400 text-2xl font-bold">
              {(confidence * 100).toFixed(0)}%
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-red-500"></div>
          <span className="text-sm text-gray-300">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>
    </div>
  );
};

export default GestureCamera;
