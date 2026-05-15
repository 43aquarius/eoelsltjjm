import { createBrowserRouter } from "react-router";
import { Layout } from "./components/Layout";
import { Home } from "./pages/Home";
import { AddRecord } from "./pages/AddRecord";
import { MemorialCard } from "./pages/MemorialCard";
import { Admin } from "./pages/Admin";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Layout,
    children: [
      { index: true, Component: Home },
      { path: "add", Component: AddRecord },
      { path: "card", Component: MemorialCard },
      { path: "admin", Component: Admin },
    ],
  },
]);
