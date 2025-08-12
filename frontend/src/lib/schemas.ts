import { z } from "zod";

export const RecommendationSchema = z.object({
  symbol: z.string(),
  rating: z.string(),
  rationale: z.string().optional(),
});

export const InsightSchema = z.object({
  title: z.string(),
  description: z.string(),
});


