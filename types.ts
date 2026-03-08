
export type Studio = 'photo-video' | 'product-launch' | 'live' | 'post' | 'analytic' | 'guide' | 'history';

export interface LiveProduct {
  id: number;
  name: string;
  price: string;
  info: string;
  features: string;
  benefits: string;
}

export interface TimelineSegment {
  time: string;
  title: string;
  script: string;
  icon: string;
  productIndex?: number;
}

export interface SavedPrompt {
    id: number;
    studio: Studio;
    type: string;
    content: string;
    timestamp: number;
}
