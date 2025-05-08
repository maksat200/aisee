import { useEffect, useState } from 'react'
import Navbar from '../components/Navbar'
import ModalNotAuth from '../components/ModalNotAuth'
import ProfilebackgroundImg from '../assets/profile_bg_new.jpeg'
export default function Profile() {
  const [profile, setProfile] = useState(null)
  const [showModal, setShowModal] = useState(false)

  useEffect(() => {
    const token = localStorage.getItem('token')
    if (!token) {
      setShowModal(true)
      return
    }

    fetch('http://localhost:8000/user/me', {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
      .then(res => {
        if (!res.ok) throw new Error('Unauthorized')
        return res.json()
      })
      .then(data => setProfile(data))
      .catch(err => {
        console.error('Not authorized', err)
        setShowModal(true)
      })
  }, [])

  return (
    <>
      <Navbar />
      {showModal && <ModalNotAuth onClose={() => setShowModal(false)} />}

      {profile && (
        <div className="min-h-screen bg-[#fef7f1] flex flex-col items-center pb-16 px-4">
          {/* Контейнер хедера + инфо */}
          <div className="w-full flex justify-center">
            <div className="w-full max-w-3xl relative">
              {/* Оранжевый блок */}
              <div className="h-40 bg-gradient-to-r from-orange-300 to-orange-400 rounded-b-2xl shadow-md"  style={{ backgroundImage: `url(${ProfilebackgroundImg})` }} />

              {/* Фото и надпись */}
              <div className="absolute bottom-0 left-6 flex items-center gap-6">
                <img
                  src="https://randomuser.me/api/portraits/women/44.jpg"
                  alt="Avatar"
                  className="w-28 h-28 rounded-full border-4 border-white shadow-md bg-white object-cover"
                />
                <div className="px-4 py-2 bg-[#d5a67c] text-white text-sm rounded-full shadow-md">
                  Subscription type: <strong>Basic</strong>
                </div>
              </div>
            </div>
          </div>

          {/* Форма профиля */}
          <div className="mt-16 w-full max-w-3xl bg-white rounded-2xl shadow-md p-8 grid grid-cols-1 sm:grid-cols-2 gap-6 text-gray-800">
            <div className="flex flex-col">
              <label className="mb-1 font-semibold">First name</label>
              <input
                type="text"
                placeholder="First name"
                className="border border-gray-300 rounded px-3 py-2 outline-orange-300"
                defaultValue={profile.first_name}
              />
            </div>
            <div className="flex flex-col">
              <label className="mb-1 font-semibold">Last name</label>
              <input
                type="text"
                placeholder="Last name"
                className="border border-gray-300 rounded px-3 py-2 outline-orange-300"
                defaultValue={profile.last_name}
              />
            </div>
            <div className="flex flex-col">
              <label className="mb-1 font-semibold">Email</label>
              <input
                type="email"
                placeholder="Email"
                className="border border-gray-300 rounded px-3 py-2 outline-orange-300"
                defaultValue={profile.email}
              />
            </div>
            <div className="flex flex-col">
              <label className="mb-1 font-semibold">Password</label>
              <input
                type="password"
                placeholder="Password"
                className="border border-gray-300 rounded px-3 py-2 outline-orange-300"
              />
            </div>
            <div className="flex flex-col">
              <label className="mb-1 font-semibold">Telegram ID</label>
              <input
                type="text"
                placeholder="Telegram ID"
                className="border border-gray-300 rounded px-3 py-2 outline-orange-300"
              />
            </div>
            <div className="flex flex-col">
              <label className="mb-1 font-semibold">Pick affiliate</label>
              <input
                type="text"
                placeholder="Affiliate"
                className="border border-gray-300 rounded px-3 py-2 outline-orange-300"
              />
            </div>
          </div>
        </div>
      )}
    </>
  )
}
