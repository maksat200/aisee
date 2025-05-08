import { useNavigate } from 'react-router-dom'

export default function ModalNotAuth() {
  const navigate = useNavigate()

  return (
    <div className="fixed inset-0 bg-black bg-opacity-40 flex justify-center items-center z-50">
      <div className="bg-white p-6 rounded-xl text-center shadow-lg">
        <h2 className="text-xl font-semibold mb-4">Вы не авторизованы</h2>
        <p className="mb-4">Пожалуйста, залогиньтесь или зарегистрируйтесь</p>

        <button
          onClick={() => navigate('/login')}
          className="bg-orange-400 text-white px-4 py-2 rounded mb-2 w-full"
        >
          Логин / Регистрация
        </button>

        <button
          onClick={() => navigate('/')}
          className="text-gray-500 hover:text-gray-800 text-sm"
        >
          Отмена
        </button>
      </div>
    </div>
  )
}
