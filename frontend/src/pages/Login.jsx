import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Navbar from '../components/Navbar'

export default function Login() {
  const [isLogin, setIsLogin] = useState(true)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [firstName, setFirstName] = useState('')
  const [lastName, setLastName] = useState('')
  const [error, setError] = useState('')
  const navigate = useNavigate()

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')

    const endpoint = isLogin
      ? 'http://127.0.0.1:8000/auth/token'
      : 'http://127.0.0.1:8000/user/register'

    try {
      let res

      if (isLogin) {
        // Авторизация: application/x-www-form-urlencoded
        const formData = new URLSearchParams()
        formData.append('username', email)
        formData.append('password', password)

        res = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: formData.toString()
        })
      } else {
        // Регистрация: application/json
        const body = {
          email,
          password,
          first_name: firstName,
          last_name: lastName,
          is_paid: false,
          is_superuser: false,
          subscription_type: 'basic',
          other_data: {}
        }

        res = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        })
      }

      const contentType = res.headers.get('content-type')
      const data = contentType?.includes('application/json')
        ? await res.json()
        : await res.text()

      if (!res.ok) {
        const errorMessage =
          typeof data === 'string'
            ? data
            : data?.detail || 'Что-то пошло не так'
        throw new Error(errorMessage)
      }

      if (isLogin) {
        localStorage.setItem('token', data.access_token)
        navigate('/profile')
      } else {
        setIsLogin(true)
      }
    } catch (err) {
      setError(err.message || 'Ошибка при подключении к серверу')
    }
  }

  return (
    <>
      <Navbar />
      <div className="min-h-screen flex justify-center items-center bg-orange-50">
        <form
          onSubmit={handleSubmit}
          className="bg-white p-8 rounded-xl shadow-md w-[360px] space-y-6 transition-all transform hover:scale-105 hover:shadow-xl"
        >
          <div className="flex justify-center gap-6 mb-6">
            <button
              type="button"
              onClick={() => setIsLogin(true)}
              className={`font-semibold text-lg ${
                isLogin ? 'text-orange-500 underline' : 'text-gray-500'
              } transition-all duration-300`}
            >
              Login
            </button>
            <span className="text-gray-500">/</span>
            <button
              type="button"
              onClick={() => setIsLogin(false)}
              className={`font-semibold text-lg ${
                !isLogin ? 'text-orange-500 underline' : 'text-gray-500'
              } transition-all duration-300`}
            >
              Registration
            </button>
          </div>

          {error && <p className="text-red-500 text-sm text-center">{error}</p>}

          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full p-3 border rounded-xl focus:outline-none focus:ring-2 focus:ring-orange-400 transition-all"
            required
          />

          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full p-3 border rounded-xl focus:outline-none focus:ring-2 focus:ring-orange-400 transition-all"
            required
          />

          {!isLogin && (
            <>
              <input
                type="text"
                placeholder="First Name"
                value={firstName}
                onChange={(e) => setFirstName(e.target.value)}
                className="w-full p-3 border rounded-xl focus:outline-none focus:ring-2 focus:ring-orange-400 transition-all"
                required
              />
              <input
                type="text"
                placeholder="Last Name"
                value={lastName}
                onChange={(e) => setLastName(e.target.value)}
                className="w-full p-3 border rounded-xl focus:outline-none focus:ring-2 focus:ring-orange-400 transition-all"
                required
              />
            </>
          )}

          <button
            type="submit"
            className="w-full bg-orange-500 text-white py-3 rounded-xl hover:bg-orange-600 transition duration-300 transform hover:scale-105"
          >
            {isLogin ? 'Sign In' : 'Register'}
          </button>
        </form>
      </div>
    </>
  )
}
