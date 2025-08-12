export default function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded border border-red-300 bg-red-50 p-3 text-red-800">
      {message}
    </div>
  );
}


