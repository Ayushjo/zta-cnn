import { BrowserRouter, Route, Routes } from 'react-router'
import Layout from '@/components/Layout'
import Checker from '@/pages/Checker'
import Demo from '@/pages/Demo'
import Design from '@/pages/Design'
import Hardware from '@/pages/Hardware'
import Overview from '@/pages/Overview'
import Present from '@/pages/Present'
import Quantization from '@/pages/Quantization'
import Results from '@/pages/Results'

export default function App() {
  return (
    <BrowserRouter basename={import.meta.env.BASE_URL.replace(/\/$/, '') || undefined}>
      <Routes>
        <Route path="/present" element={<Present />} />
        <Route element={<Layout />}>
          <Route index element={<Overview />} />
          <Route path="demo" element={<Demo />} />
          <Route path="design" element={<Design />} />
          <Route path="quantization" element={<Quantization />} />
          <Route path="checker" element={<Checker />} />
          <Route path="results" element={<Results />} />
          <Route path="hardware" element={<Hardware />} />
          <Route path="*" element={<Overview />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
