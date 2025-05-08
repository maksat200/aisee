import { useNavigate } from 'react-router-dom'
import Navbar from '../components/Navbar'
import cameraImg from '../assets/camera.png'

export default function Home() {
  const navigate = useNavigate()

  const handleStart = () => {
    navigate('/login') // или '/register' если нужно
  }

  const handleBuyNow = () => {
    navigate('/login')
  }

  return (
    <>
      <Navbar />
      <div className="min-h-screen bg-white font-sans">

        <section className="w-full bg-white px-6 py-12 flex flex-col md:flex-row items-center justify-center overflow-hidden">
          <div className="md:w-2/3 bg-primary text-white px-10 py-12 clip-left-reverse relative z-10">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Aisee - Your View on Business</h2>
            <p className="text-sm md:text-base mb-6">
              Smart Video Analytics for Cafes and Restaurants: Control, Efficiency, Profit
            </p>
            <button
              onClick={handleStart}
              className="bg-white text-primary font-semibold py-2 px-6 rounded-full shadow hover:bg-gray-100 transition"
            >
              Get started
            </button>
          </div>

          <div className="md:w-1/3 mt-8 md:mt-0 -ml-6 relative z-0 flex justify-center">
            <div className="w-full h-[300px] rounded-[40px] overflow-hidden shadow-xl bg-white p-2 clip-camera-corner">
              <img src={cameraImg} alt="camera" className="w-full h-full object-cover" />
            </div>
          </div>
        </section>

        <section className="text-center py-12 bg-white">
          <h3 className="text-2xl font-semibold mb-2">How it helps to your business?</h3>
          <p className="text-gray-600 max-w-xl mx-auto px-4">
            Aisee analyzes how many customers visit your café or restaurant. It helps you decide where to add or reduce waiters.
            Aisee also tracks staff activity, like how long a chef is away from the kitchen. This improves service, reduces costs, and increases profits.
          </p>
        </section>

        <section className="py-12 bg-[#FFF1DE] flex justify-center">
          <div className="bg-[#E0AD7D] text-white w-80 p-8 rounded-[40px] text-center shadow-md relative">
            <div className="text-xl font-medium mb-2">Subscription</div>
            <div className="absolute top-6 right-6 w-16 h-16 bg-[#F1D3B6] text-[#5C504A] rounded-full flex items-center justify-center text-lg font-semibold shadow-sm">
              100$
            </div>
            <hr className="border-t border-white/70 my-4" />
            <div className="text-lg font-light space-y-2 mb-6">
              <p>Staff analytics</p>
              <p>Counts customers</p>
            </div>
            <button
              onClick={handleBuyNow}
              className="bg-[#F1D3B6] text-[#5C504A] text-lg px-6 py-2 rounded-full shadow-md hover:scale-105 transition"
            >
              Buy now!
            </button>
          </div>
        </section>

        <section className="text-center py-12 bg-gray-100">
          <h3 className="text-xl font-semibold mb-2">AISEE will be helping cafes and restaurants</h3>
          <p className="text-sm text-gray-600 mb-4">such as</p>
          <div className="flex justify-center gap-12 flex-wrap">
            <img src="/src/assets/capito.png" alt="capito" className="w-24 h-24 object-contain rounded-full shadow" />
            <img src="/src/assets/girrafe.png" alt="girrafe" className="w-24 h-24 object-contain rounded-full shadow" />
          </div>
        </section>

        <section className="text-center py-12">
          <h3 className="text-xl font-semibold mb-4">See your analytics</h3>
          <p className="text-sm text-gray-500 mb-6">in profile</p>
          <div className="flex justify-center gap-12 px-4">
            <div className="w-40 h-24 bg-gray-100 shadow rounded flex items-center justify-center">Staff analytics</div>
            <div className="w-40 h-24 bg-gray-100 shadow rounded flex items-center justify-center">Customer analytics</div>
          </div>
        </section>

        <footer className="bg-orange-100 text-center py-6 text-sm text-gray-600">
          <p>© AISEE | aisee.project.site</p>
          <p>contact@aisee.com</p>
        </footer>

      </div>
    </>
  )
}
