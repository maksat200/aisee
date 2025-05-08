import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import Upload from './pages/Upload'
import Profile from './pages/Profile'
import Login from './pages/Login'
import Analytics from './pages/Analytics'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/profile" element={<Profile />} />
        <Route path="/login" element={<Login />} />
        <Route path="/analytics" element={<Analytics />} />
      </Routes>
    </BrowserRouter>
  )
}
