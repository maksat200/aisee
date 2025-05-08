import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Navbar from '../components/Navbar'
import backgroundImg from '../assets/background.png'

export default function Upload() {
  const [videoFile, setVideoFile] = useState(null)
  const [message, setMessage] = useState('')
  const [uploadSuccess, setUploadSuccess] = useState(false)
  const navigate = useNavigate()

  const handleChange = (e) => {
    setVideoFile(e.target.files[0])
    setMessage('')
    setUploadSuccess(false)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!videoFile) return setMessage('❗ Please select a video file')

    const formData = new FormData()
    formData.append('file', videoFile)

    try {
      setMessage('⏳ Uploading...')
      setTimeout(() => {
        setUploadSuccess(true)
        setMessage('Upload success!')
      }, 1500)
    } catch (err) {
      setMessage('Upload error: ' + err.message)
    }
  }

  return (
    <>
      <Navbar />
      <div
        className="min-h-screen flex items-center justify-center bg-cover bg-no-repeat bg-center px-4"
        style={{ backgroundImage: `url(${backgroundImg})` }}
      >
        <form
          onSubmit={handleSubmit}
          className="w-full max-w-md p-8 bg-white/90 backdrop-blur-sm rounded-3xl shadow-2xl text-center flex flex-col items-center gap-5"
        >
          <h2 className="text-3xl font-bold text-gray-800">Upload Your Video</h2>
          <p className="text-sm text-gray-500 leading-tight">
            Drag and drop your file here<br />or choose manually:
          </p>

          <label className="bg-[#e0ad7d] text-white px-6 py-2 rounded-xl cursor-pointer hover:bg-opacity-90 transition">
            Select file
            <input type="file" accept="video/*" onChange={handleChange} hidden />
          </label>

          <button
            type="submit"
            className="px-6 py-2 bg-orange-400 text-white rounded-xl hover:bg-orange-500 transition"
          >
            Upload
          </button>

          {message && <p className="text-sm text-gray-700">{message}</p>}

          {uploadSuccess && (
            <button
              onClick={() => navigate('/analytics')}
              className="px-6 py-2 bg-orange-400 text-white rounded-xl hover:bg-orange-500 transition"
            >
              Go to Analytics
            </button>
          )}
        </form>
      </div>
    </>
  )
}
