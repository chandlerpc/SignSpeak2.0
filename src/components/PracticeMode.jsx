import React, { useState, useEffect, useRef } from 'react'
import { Hands } from '@mediapipe/hands'
import { Camera } from '@mediapipe/camera_utils'
import './PracticeMode.css'

const PracticeMode = () => {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [targetLetter, setTargetLetter] = useState('A')
  const [userAttempt, setUserAttempt] = useState(null)
  const [score, setScore] = useState(0)
  const [totalAttempts, setTotalAttempts] = useState(0)
  const [feedback, setFeedback] = useState('')
  const [practiceHistory, setPracticeHistory] = useState([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [showCheatSheet, setShowCheatSheet] = useState(false)
  const [isTrackingEnabled, setIsTrackingEnabled] = useState(false)
  const isTrackingEnabledRef = useRef(false)
  const [isDetecting, setIsDetecting] = useState(false)
  const [fps, setFps] = useState(0)
  const lastFrameTime = useRef(Date.now())
  const frameCount = useRef(0)
  const lastPredictionTime = useRef(0)

  // Keep ref in sync with state
  useEffect(() => {
    isTrackingEnabledRef.current = isTrackingEnabled
  }, [isTrackingEnabled])
  useEffect(() => {
    initializeCamera()
    generateRandomLetter()
  }, [])

  const initializeCamera = async () => {
    try {
      const hands = new Hands({
        locateFile: (file) => {
          return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        }
      })

      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.5
      })

      hands.onResults(onHandsResults)

      if (videoRef.current) {
        const camera = new Camera(videoRef.current, {
          onFrame: async () => {
            await hands.send({ image: videoRef.current })
          },
          width: 640,
          height: 480
        })
        camera.start()
      }
    } catch (error) {
      console.error('Error initializing camera:', error)
    }
  }

  const onHandsResults = async (results) => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')

    canvas.width = results.image.width
    canvas.height = results.image.height

    ctx.save()
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height)

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      setIsDetecting(true)
      const landmarks = results.multiHandLandmarks[0]
      drawHandLandmarks(ctx, landmarks, canvas.width, canvas.height)

      // Throttle predictions to every 1 second, only if tracking is enabled
      if (isTrackingEnabledRef.current) {
        const now = Date.now()
        if (now - lastPredictionTime.current > 1000 && !isProcessing) {
          lastPredictionTime.current = now
          await classifyAndCheck(landmarks, canvas.width, canvas.height, ctx)
        }
      }
    } else {
      setIsDetecting(false)
    }

    ctx.restore()
    updateFPS()
  }

  const updateFPS = () => {
    frameCount.current++
    const now = Date.now()
    const elapsed = now - lastFrameTime.current

    if (elapsed >= 1000) {
      setFps(frameCount.current)
      frameCount.current = 0
      lastFrameTime.current = now
    }
  }

  const drawHandLandmarks = (ctx, landmarks, width, height) => {
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4],
      [0, 5], [5, 6], [6, 7], [7, 8],
      [0, 9], [9, 10], [10, 11], [11, 12],
      [0, 13], [13, 14], [14, 15], [15, 16],
      [0, 17], [17, 18], [18, 19], [19, 20],
      [5, 9], [9, 13], [13, 17]
    ]

    ctx.strokeStyle = '#00ff00'
    ctx.lineWidth = 2
    connections.forEach(([start, end]) => {
      ctx.beginPath()
      ctx.moveTo(landmarks[start].x * width, landmarks[start].y * height)
      ctx.lineTo(landmarks[end].x * width, landmarks[end].y * height)
      ctx.stroke()
    })

    ctx.fillStyle = '#ff0000'
    landmarks.forEach(landmark => {
      ctx.beginPath()
      ctx.arc(landmark.x * width, landmark.y * height, 5, 0, 2 * Math.PI)
      ctx.fill()
    })
  }

  const classifyAndCheck = async (landmarks, width, height, ctx) => {
    try {
      setIsProcessing(true)

      // Calculate bounding box
      const xs = landmarks.map(l => l.x * width)
      const ys = landmarks.map(l => l.y * height)
      const minX = Math.max(0, Math.min(...xs) - 30)
      const minY = Math.max(0, Math.min(...ys) - 30)
      const maxX = Math.min(width, Math.max(...xs) + 30)
      const maxY = Math.min(height, Math.max(...ys) + 30)

      // Extract ROI
      const roi = ctx.getImageData(minX, minY, maxX - minX, maxY - minY)

      // Create temporary canvas for resizing
      const tempCanvas = document.createElement('canvas')
      tempCanvas.width = maxX - minX
      tempCanvas.height = maxY - minY
      const tempCtx = tempCanvas.getContext('2d')
      tempCtx.putImageData(roi, 0, 0)

      // Create final canvas at 128x128 (matches model input size)
      const finalCanvas = document.createElement('canvas')
      finalCanvas.width = 128
      finalCanvas.height = 128
      const finalCtx = finalCanvas.getContext('2d')
      finalCtx.drawImage(tempCanvas, 0, 0, 128, 128)

      // Get pixel data and normalize
      const imageData = finalCtx.getImageData(0, 0, 128, 128)
      const pixels = imageData.data

      // Convert RGBA to RGB and normalize to [0, 1]
      const normalized = []
      for (let i = 0; i < pixels.length; i += 4) {
        normalized.push(pixels[i] / 255.0)     // R
        normalized.push(pixels[i + 1] / 255.0) // G
        normalized.push(pixels[i + 2] / 255.0) // B
      }

      // Reshape to [128, 128, 3]
      const normalizedImage = []
      for (let y = 0; y < 128; y++) {
        const row = []
        for (let x = 0; x < 128; x++) {
          const idx = (y * 128 + x) * 3
          row.push([normalized[idx], normalized[idx + 1], normalized[idx + 2]])
        }
        normalizedImage.push(row)
      }

      // Send to Flask API
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: normalizedImage }),
      })

      if (response.ok) {
        const result = await response.json()
        if (result.prediction && result.confidence > 0.5) {
          // Only set if it's a valid letter (not 'del', 'nothing', 'space')
          const predictedLetter = result.prediction.toUpperCase()
          if (predictedLetter.length === 1 && /[A-Z]/.test(predictedLetter)) {
            setUserAttempt({ letter: predictedLetter, confidence: result.confidence })
          }
        }
      }
    } catch (error) {
      console.error('Classification error:', error)
    } finally {
      setIsProcessing(false)
    }
  }

  const generateRandomLetter = () => {
    const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    const randomLetter = letters[Math.floor(Math.random() * letters.length)]
    setTargetLetter(randomLetter)
    setUserAttempt(null)
    setFeedback('')
  }

  const checkAnswer = () => {
    if (!userAttempt) {
      setFeedback('No sign detected. Please try again.')
      return
    }

    const isCorrect = userAttempt.letter === targetLetter
    setTotalAttempts(prev => prev + 1)

    if (isCorrect) {
      setScore(prev => prev + 1)
      setFeedback(`Correct! Well done! Confidence: ${(userAttempt.confidence * 100).toFixed(0)}%`)
      setPracticeHistory(prev => [...prev, { target: targetLetter, result: 'correct', confidence: userAttempt.confidence, timestamp: Date.now() }])

      setTimeout(() => {
        generateRandomLetter()
      }, 2000)
    } else {
      setFeedback(`Incorrect. You signed "${userAttempt.letter}" but the target was "${targetLetter}". Try again!`)
      setPracticeHistory(prev => [...prev, { target: targetLetter, attempted: userAttempt.letter, result: 'incorrect', confidence: userAttempt.confidence, timestamp: Date.now() }])
    }
  }

  const resetPractice = () => {
    setScore(0)
    setTotalAttempts(0)
    setPracticeHistory([])
    generateRandomLetter()
  }

  const accuracy = totalAttempts > 0 ? ((score / totalAttempts) * 100).toFixed(1) : 0

  return (
    <div className="practice-mode">
      <div className="practice-header">
        <h2>Practice Mode</h2>
        <div className="practice-stats">
          <div className="stat-item">
            <span className="stat-label">Score:</span>
            <span className="stat-value">{score}/{totalAttempts}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Accuracy:</span>
            <span className="stat-value">{accuracy}%</span>
          </div>
        </div>
      </div>

      <div className="practice-content">
        <div className="practice-camera">
          <div className="camera-header">
            <h2>Camera Feed</h2>
            <div className="camera-controls">
              <button
                className={`tracking-toggle ${isTrackingEnabled ? 'active' : ''}`}
                onClick={() => setIsTrackingEnabled(!isTrackingEnabled)}
              >
                {isTrackingEnabled ? '⏸ Pause Tracking' : '▶ Start Tracking'}
              </button>
            </div>
          </div>
          <div className="camera-container">
            <video ref={videoRef} style={{ display: 'none' }} />
            <canvas ref={canvasRef} className="practice-canvas" />
          </div>
          <div className="camera-status">
            <div className={`status-indicator ${isDetecting ? 'detected' : ''}`}>
              <span className="status-dot"></span>
              {isDetecting ? 'Hand Detected' : 'No Hand Detected'}
            </div>
            <div className="fps-counter">
              FPS: {fps}
            </div>
          </div>
        </div>

        <div className="practice-panel">
          <div className="target-display">
            <h3>Sign this letter:</h3>
            <div className="target-letter">{targetLetter}</div>
          </div>

          {userAttempt && (
            <div className="attempt-display">
              <h4>Your Sign:</h4>
              <div className="attempt-letter">{userAttempt.letter}</div>
              <div className="attempt-confidence">
                Confidence: {(userAttempt.confidence * 100).toFixed(0)}%
              </div>
            </div>
          )}

          {feedback && (
            <div className={`feedback ${feedback.includes('Correct') ? 'success' : 'error'}`}>
              {feedback}
            </div>
          )}

          <div className="practice-controls">
            <button className="practice-btn check" onClick={checkAnswer}>
              Check Answer
            </button>
            <button className="practice-btn skip" onClick={generateRandomLetter}>
              Skip Letter
            </button>
            <button className="practice-btn cheat-sheet" onClick={() => setShowCheatSheet(!showCheatSheet)}>
              {showCheatSheet ? 'Hide' : 'Show'} Cheat Sheet
            </button>
            <button className="practice-btn reset" onClick={resetPractice}>
              Reset Practice
            </button>
          </div>

          <div className="recent-attempts">
            <h4>Recent Attempts</h4>
            <div className="attempts-list">
              {practiceHistory.slice(-5).reverse().map((item, i) => (
                <div key={i} className={`attempt-item ${item.result}`}>
                  <span className="attempt-target">{item.target}</span>
                  {item.attempted && <span className="attempt-arrow">→</span>}
                  {item.attempted && <span className="attempt-attempted">{item.attempted}</span>}
                  <span className={`attempt-result ${item.result}`}>
                    {item.result === 'correct' ? '✓' : '✗'}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {showCheatSheet && (
            <div className="cheat-sheet">
              <h4>ASL Alphabet Reference</h4>
              <img
                src="/ASL-Alphabet-poster-flashcards-683x1024.png"
                alt="ASL Alphabet Reference"
                className="cheat-sheet-image"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default PracticeMode
