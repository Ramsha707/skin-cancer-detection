import { lazy, Suspense } from "react";
import { BrowserRouter, Navigate, Outlet, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AppLayout } from "@/components/layout/AppLayout";
import { PageLoader } from "@/components/common/PageLoader";

import Landing from "@/pages/Landing";

/* Each week's page is code-split into its own chunk. */
const Dashboard = lazy(() => import("@/pages/Dashboard"));
const Agents = lazy(() => import("@/pages/Agents"));
const AgentDetail = lazy(() => import("@/pages/AgentDetail"));
const Detection = lazy(() => import("@/pages/Detection"));
const FederatedTraining = lazy(() => import("@/pages/FederatedTraining"));
const PretrainedModel = lazy(() => import("@/pages/PretrainedModel"));
const Experiments = lazy(() => import("@/pages/Experiments"));
const ModelRegistry = lazy(() => import("@/pages/ModelRegistry"));
const Architecture = lazy(() => import("@/pages/Architecture"));

/* Week 9-12 routes are deliberately stubbed rather than faked. */
const Performance = lazy(() => import("@/pages/stubs/Performance"));
const Explainability = lazy(() => import("@/pages/stubs/Explainability"));
const Privacy = lazy(() => import("@/pages/stubs/Privacy"));
const Audit = lazy(() => import("@/pages/stubs/Audit"));
const Timeline = lazy(() => import("@/pages/stubs/Timeline"));
const Settings = lazy(() => import("@/pages/stubs/Settings"));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 2000,
    },
  },
});

/** Shared suspense boundary for every lazily-loaded console route. */
function ConsoleRoutes() {
  return (
    <AppLayout>
      <Suspense fallback={<PageLoader />}>
        <Outlet />
      </Suspense>
    </AppLayout>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          {/* Marketing surface sits outside the console shell */}
          <Route path="/" element={<Landing />} />

          <Route element={<ConsoleRoutes />}>
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/agents" element={<Agents />} />
            <Route path="/agents/:id" element={<AgentDetail />} />
            <Route path="/detection" element={<Detection />} />
            <Route path="/federated-training" element={<FederatedTraining />} />
            <Route path="/architecture" element={<Architecture />} />
            <Route path="/experiments" element={<Experiments />} />
            <Route path="/models" element={<ModelRegistry />} />
            <Route path="/models/pretrained" element={<PretrainedModel />} />

            <Route path="/performance" element={<Performance />} />
            <Route path="/explainability" element={<Explainability />} />
            <Route path="/privacy" element={<Privacy />} />
            <Route path="/audit" element={<Audit />} />
            <Route path="/timeline" element={<Timeline />} />
            <Route path="/settings" element={<Settings />} />
          </Route>

          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}