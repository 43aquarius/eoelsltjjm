import { RouterProvider } from 'react-router';
import { Toaster } from 'sonner';
import { router } from './routes';
import { StoreProvider } from './store';

function App() {
  return (
    <StoreProvider>
      <RouterProvider router={router} />
      <Toaster position="top-center" richColors />
    </StoreProvider>
  );
}

export default App;
