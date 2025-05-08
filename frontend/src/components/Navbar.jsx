import { Link } from 'react-router-dom'
import logo from '../assets/favicon.svg'

export default function Navbar() {
  return (
    <nav className="bg-orange-100 px-6 py-4 flex justify-between items-center shadow-sm">
      <div className="flex items-center gap-2">
        <img src={logo} alt="logo" className="w-6 h-6" />
        <Link to="/" className="text-xl font-bold text-orange-500">AISEE</Link>
      </div>
      <div className="flex items-center gap-6 text-sm text-gray-700">
        <Link to="/" className="hover:text-orange-500">Home</Link>
        <Link to="/upload" className="hover:text-orange-500">Upload</Link>
        <Link to="/profile" className="hover:text-orange-500">Profile</Link>
        <Link to="/analytics" className="hover:text-orange-500">Analytics</Link>
        <Link to="/login" className="px-3 py-1 border border-orange-400 text-orange-500 rounded hover:bg-orange-100">Login</Link>
      </div>
    </nav>
  )
}
